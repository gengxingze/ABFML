"""Independent ACSF formulas, angular pairs, derivatives, configuration, and calibration tests."""

from abfml.data.reader import from_ase

import math
import json
import tempfile
import unittest
from dataclasses import replace
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np
import torch
from ase import Atoms

from abfml.core.registry import build_model, parse_options
from abfml.builtin.acsf import AngularBasis, RadialBasis, ACSFAngularDescriptor, ACSFRadialDescriptor
from abfml.config import ConfigError
from abfml.core import AtomicSample, ModelInput
from abfml.core.neighbor import ASENeighborBuilder
from abfml.data import StructureCollator
from abfml.interfaces import ABFMLCalculator


def acsf_config():
    return {"name": "acsf", "descriptor": {"cutoff": 3., "max_neighbors": 8,
        "cutoff_function": "cos", "blocks": [
            {"type": "radial_gaussian", "basis": [{"eta": .2, "rs": .1}, {"eta": .4, "rs": .5}]},
            {"type": "angular_narrow", "basis": [{"eta": .1, "rs": .3, "lambda": 1, "zeta": 2}]},
            {"type": "angular_wide", "basis": [{"eta": .2, "rs": 0., "lambda": -1, "zeta": 1}]},
        ]}, "fitting": {"hidden_layers": [5, 5], "activation": "tanh", "residual": True}}


def make_model(config=None, type_map=(1, 8)):
    # Construct parameters in the target dtype to avoid initial float32 rounding of basis constants.
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        parsed = parse_options(config or acsf_config(), type_map=type_map)
        return build_model("acsf", **parsed.model_kwargs()).eval()
    finally:
        torch.set_default_dtype(previous)


class ACSFDescriptorTest(unittest.TestCase):
    def setUp(self):
        self.previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        self.addCleanup(torch.set_default_dtype, self.previous_dtype)
        self.vectors = torch.tensor([[[.8,.2,.1], [.1,1.1,.3], [-.4,.2,1.2], [0.,0.,0.]]])
        self.types = torch.tensor([[0, 1, 0, 1]])
        self.valid = torch.tensor([[True, True, True, False]])
        self.basis = (AngularBasis(.1,.3,1.,2), AngularBasis(.2,0.,-1.,1))

    def explicit_angular(self, include_rjk):
        """Independent scalar reference using real j<k pairs, without reusing the tested tensor
        path.
        """
        pairs = list(combinations_with_replacement(range(2), 2))
        result = np.zeros((len(pairs), len(self.basis)))
        vectors = self.vectors[0].numpy()
        def fc(r):
            return .5 * (math.cos(math.pi*r/3.) + 1.) if r < 3. else 0.
        for j, k in combinations(range(3), 2):
            rj, rk = np.linalg.norm(vectors[j]), np.linalg.norm(vectors[k])
            rjk = np.linalg.norm(vectors[j]-vectors[k])
            cosine = np.dot(vectors[j],vectors[k])/(rj*rk)
            channel = pairs.index(tuple(sorted((int(self.types[0,j]),int(self.types[0,k])))))
            for b, basis in enumerate(self.basis):
                exponent = (rj-basis.rs)**2 + (rk-basis.rs)**2
                envelope = fc(rj)*fc(rk)
                if include_rjk:
                    exponent += (rjk-basis.rs)**2
                    envelope *= fc(rjk)
                result[channel,b] += (2**(1-basis.zeta) * (1+basis.lambd*cosine)**basis.zeta
                                      * math.exp(-basis.eta*exponent) * envelope)
        return torch.tensor(result.reshape(1,-1))

    def test_angular_matches_independent_formula(self):
        for narrow in (False,True):
            with self.subTest(narrow=narrow):
                block = ACSFAngularDescriptor(2,3.,self.basis,include_rjk=narrow)
                torch.testing.assert_close(block(self.types,self.vectors,self.valid),
                                           self.explicit_angular(narrow),rtol=1.e-12,atol=1.e-12)

    def test_radial_formula_and_element_channels(self):
        basis = (RadialBasis(.2,.3), RadialBasis(.4,1.))
        block = ACSFRadialDescriptor(2,3.,basis)
        expected = np.zeros((2,2))
        for j in range(3):
            r = np.linalg.norm(self.vectors[0,j].numpy())
            for b,p in enumerate(basis):
                expected[self.types[0,j],b] += math.exp(-p.eta*(r-p.rs)**2)*.5*(math.cos(math.pi*r/3.)+1)
        torch.testing.assert_close(block(self.types,self.vectors,self.valid),torch.tensor(expected.reshape(1,-1)))

    def test_single_neighbor_has_no_angular_contribution(self):
        for narrow in (False,True):
            block = ACSFAngularDescriptor(2,3.,self.basis,include_rjk=narrow)
            # Cover no neighbor pairs and a single valid slot surrounded by padding.
            for width in (1,4):
                mask = torch.zeros_like(self.valid[:,:width])
                mask[:,0] = True
                output = block(self.types[:,:width],self.vectors[:,:width],mask)
                torch.testing.assert_close(output,torch.zeros(1,6))

    def test_permutation_and_padding_invariance(self):
        permutation = torch.tensor([2,0,3,1])
        for narrow in (False,True):
            block = ACSFAngularDescriptor(2,3.,self.basis,include_rjk=narrow)
            expected = block(self.types,self.vectors,self.valid)
            actual = block(self.types[:,permutation],self.vectors[:,permutation],self.valid[:,permutation])
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(block(self.types[:,:3],self.vectors[:,:3],self.valid[:,:3]),expected)

    def test_narrow_and_wide_have_distinct_support(self):
        vectors = torch.tensor([[[2.4,0.,0.],[-2.4,0.,0.]]])
        types,valid = torch.zeros(1,2,dtype=torch.long),torch.ones(1,2,dtype=torch.bool)
        basis=(AngularBasis(.1,0.,-1.,1),)
        narrow=ACSFAngularDescriptor(1,3.,basis,include_rjk=True)
        wide=ACSFAngularDescriptor(1,3.,basis,include_rjk=False)
        self.assertEqual(narrow(types,vectors,valid).item(),0.)
        self.assertGreater(wide(types,vectors,valid).item(),0.)

    def test_first_and_second_derivatives(self):
        for narrow in (False,True):
            block=ACSFAngularDescriptor(2,3.,self.basis,include_rjk=narrow)
            vectors=self.vectors[:,:3].clone().requires_grad_(True)
            function=lambda v: block(self.types[:,:3],v,self.valid[:,:3])
            self.assertTrue(torch.autograd.gradcheck(function,(vectors,)))
            self.assertTrue(torch.autograd.gradgradcheck(function,(vectors,)))


class ACSFModelTest(unittest.TestCase):
    def test_output_descriptors_are_normalized_fitting_inputs(self):
        model = make_model()
        model.normalizer.set_statistics(torch.full((2, 10), 2., dtype=torch.float64),
                                        torch.full((2, 10), .3, dtype=torch.float64))
        atoms = Atoms('HOH', positions=[[0., 0., 0.], [1., .2, 0.], [.2, 1., .1]])
        inputs, _ = StructureCollator(ASENeighborBuilder(model.cutoff))([from_ase(atoms)])
        raw, types = model.descriptor(inputs)
        output = model(inputs)
        descriptors = output.properties['descriptors']
        self.assertEqual(descriptors.shape, (3, 10))
        torch.testing.assert_close(descriptors, (raw - .3) / 2.)
        expected = model.fitting(descriptors, types).reshape(-1) + model.energy_reference[types]
        torch.testing.assert_close(expected, output.atomic_energies)

    def test_layout_and_parameter_ownership(self):
        model=make_model()
        self.assertEqual(model.descriptor.output_dim,10)
        self.assertEqual(len(model.descriptor.feature_layout),10)
        self.assertEqual(model.descriptor.feature_layout[5]['elements'],(1,8))
        self.assertEqual(model.fitting.network_size[0],10)
        self.assertTrue(all(name.startswith('fitting.') for name,_ in model.named_parameters()))
        self.assertIn('energy_reference',dict(model.named_buffers()))

    def test_periodic_images_of_same_atom_are_distinct_neighbors(self):
        config=acsf_config()
        config['descriptor']['blocks']=config['descriptor']['blocks'][2:]
        model=make_model(config,type_map=(1,))
        inputs=ModelInput(atomic_numbers=torch.tensor([1]),center_indices=torch.tensor([0]),
            edge_ptr=torch.tensor([0,2]),neighbor_indices=torch.tensor([0,0]),
            edge_vectors=torch.tensor([[1.,0.,0.],[-1.,0.,0.]],dtype=torch.float64))
        features,_=model.descriptor(inputs)
        self.assertGreater(features.item(),0.)
        torch.testing.assert_close(model(inputs).forces,torch.zeros(1,3,dtype=torch.float64))

    def test_isolated_atom_and_force_training(self):
        model=make_model().train()
        sample=from_ase(Atoms('H',positions=[[0,0,0]]))
        inputs,_=StructureCollator(ASENeighborBuilder(3.))([sample])
        output=model(inputs)
        torch.testing.assert_close(output.forces,torch.zeros(1,3,dtype=torch.float64))
        (output.energy.square().sum()+output.forces.square().sum()).backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_calibration_reuses_descriptor_without_consuming_rng(self):
        model=make_model()
        structures=[]
        for symbol,energy in [('H2',-2.),('O2',-4.)]:
            structures.append(replace(from_ase(Atoms(symbol,positions=[[0,0,0],[1.,0,0]])),
                                      energy=torch.tensor(energy,dtype=torch.float64)))
        descriptor=model.descriptor
        rng=torch.get_rng_state().clone()
        model.initialize_statistics(structures,ASENeighborBuilder(3.))
        self.assertIs(model.descriptor,descriptor)
        torch.testing.assert_close(rng,torch.get_rng_state())
        torch.testing.assert_close(model.energy_reference,torch.tensor([-1.,-2.],dtype=torch.float64))
        features,types=descriptor(StructureCollator(ASENeighborBuilder(3.))(structures)[0])
        self.assertTrue(torch.isfinite(model.normalizer(features,types)).all())

    def test_energy_and_force_continuity_at_cutoff(self):
        torch.manual_seed(11)
        model=make_model(type_map=(1,))
        model.normalizer.set_statistics(torch.ones_like(model.normalizer.feature_std)*.5,
                                         torch.ones_like(model.normalizer.feature_avg)*.2)
        atoms=Atoms('H3',positions=[[0,0,0],[1.,.2,0],[3.,0,0]])
        atoms.calc=ABFMLCalculator(model)
        def predict(distance):
            atoms.positions[2,0]=distance
            return atoms.get_potential_energy(),atoms.get_forces().copy()
        for step in (1.e-4,1.e-6):
            left, right=predict(3.-step),predict(3.+step)
            self.assertLess(abs(left[0]-right[0]),10*step)
            self.assertLess(np.max(np.abs(left[1]-right[1])),10*step)

    def test_acsf_resume_matches_uninterrupted_training(self):
        from tests.test_workflows import training_fixture
        from abfml.workflows.train import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, path = training_fixture(root)
            config['model'] = acsf_config()
            path.write_text(json.dumps(config), encoding='utf-8')
            full = run(path, epochs=2)
            config['output']['dir'] = 'split'
            path.write_text(json.dumps(config), encoding='utf-8')
            run(path, epochs=1)
            resumed = run(path, resume=root/'split/last.pt', epochs=1)
            for name, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], atol=0., rtol=0.)


class ACSFConfigurationTest(unittest.TestCase):
    def test_invalid_basis_and_old_interface_are_rejected(self):
        changes=[('eta',0.),('eta',float('nan')),('rs',-1.),('lambda',0.),('zeta',1.5),('zeta',float('inf'))]
        for key,value in changes:
            config=acsf_config()
            config['descriptor']['blocks'][1]['basis'][0][key]=value
            with self.subTest(key=key,value=value), self.assertRaises(ConfigError):
                parse_options(config,type_map=(1,8))
        for key in ('num_basis','eta','weighted'):
            config=acsf_config()
            config['descriptor']['blocks'][0][key]=1
            with self.assertRaises(ConfigError):
                parse_options(config,type_map=(1,))
        config=acsf_config()
        config['cutoff']=3.
        with self.assertRaises(ConfigError):
            parse_options(config,type_map=(1,))

    def test_block_types_and_nonempty_basis(self):
        for kind in ('SymFunAngN','angular','radial_polynomial'):
            config=acsf_config()
            config['descriptor']['blocks'][0]['type']=kind
            with self.assertRaises(ConfigError):
                parse_options(config,type_map=(1,))
        config=acsf_config()
        config['descriptor']['blocks'][0]['basis']=[]
        with self.assertRaises(ConfigError):
            parse_options(config,type_map=(1,))


if __name__ == '__main__':
    unittest.main()
