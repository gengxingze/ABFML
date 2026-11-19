# 本地预览与 GitHub Pages 发布

网站使用 MkDocs 和 Material 主题，从现有 Markdown 文档生成。构建文档不需要安装 PyTorch 或 ABFML 本体。

## 本地预览

在项目根目录、已激活的 Python 环境中执行：

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
```

打开终端给出的地址，当前配置为 <http://127.0.0.1:8000/ABFML/>。修改 Markdown 后浏览器自动刷新；按 `Ctrl+C` 停止预览。端口被占用时使用 `python -m mkdocs serve -a 127.0.0.1:8001`。

也可以用独立虚拟环境，仅安装文档依赖。Windows PowerShell 示例：

```powershell
py -3.11 -m venv .venv-docs
.venv-docs/Scripts/python.exe -m pip install -r requirements-docs.txt
.venv-docs/Scripts/python.exe -m mkdocs serve
```

## 构建静态网页

```bash
python -m mkdocs build --strict
```

生成目录为 `site/`，已加入 `.gitignore`，无需提交生成文件。构建会检查导航和文档链接；部署工作流执行相同命令。

网站支持中文搜索、深浅色切换、代码复制、Mermaid 流程图与数学公式。字体使用本地系统字体；公式渲染需要访问 MathJax CDN，Mermaid 图的前端渲染也可能需要网络。离线情况下仍可阅读正文与代码。

## 发布到 GitHub Pages

仓库中已提供 `.github/workflows/docs.yml`。首次启用时：

1. 将网站配置、文档及工作流提交到 GitHub 仓库默认分支。
2. 在仓库 **Settings → Pages → Build and deployment → Source** 中选择 **GitHub Actions**。
3. 在 **Actions → Documentation** 查看运行结果；也可点击 **Run workflow** 手动触发，选择默认分支。
4. 发布成功后访问 <https://gengxingze.github.io/ABFML/>。

此设置使用 GitHub 官方 Pages artifact 部署方式，不需要手工维护 `gh-pages` 分支，也不需要个人访问令牌。若已有旧发布工作流，应停用旧流程，避免两套任务同时部署同一个网站。环境保护规则可能要求在 GitHub 上批准部署。

`master` 或 `main` 的相关文件变更会触发构建，只有默认分支执行发布。Pull Request 只构建检查，不发布。手动触发也只有选择默认分支时才发布。

本地配置本身不会修改远程仓库设置；只有上传工作流并满足 Pages 设置后才会更新线上网站。

## 修改文档

| 文件 | 用途 |
| --- | --- |
| `mkdocs.yml` | 网站信息、导航、主题和 Markdown 扩展 |
| `requirements-docs.txt` | 文档构建依赖版本 |
| `docs/index.md` | 网站首页 |
| `docs/*.md` | 用户手册和专题文档 |
| `docs/hooks.py` | 构建时复用仓库外部文档，并转换链接 |
| `docs/assets/` | 样式及公式渲染配置 |
| `.github/workflows/docs.yml` | 自动构建与 GitHub Pages 发布 |

工具说明来自 `abfml/tools/README.md`，LAMMPS 说明来自 `lmp/ABFML/README.md`，示例索引来自 `examples/README.md`。构建时直接读取这些源文件，不在 `docs/` 维护重复副本；预览会监控它们的修改。

页面间链接在构建时转换为网站链接，指向示例配置和源代码的链接转换为 GitHub 链接。普通 Markdown 在仓库中仍可直接阅读。新增页面后请将其加入 `mkdocs.yml` 的导航，并重新执行严格构建。

安装、训练、评估、自定义模型和部分理论专题页由 `docs/hooks.py` 从完整手册或理论文档中提取，不维护重复文本。修改章节标题时需同步更新 `CHAPTERS` 映射；修改构建钩子后请重启本地预览进程。

## 常见问题

| 现象 | 处理方式 |
| --- | --- |
| 找不到 `mkdocs` | 用当前 Python 安装 `requirements-docs.txt`，再执行 `python -m mkdocs` |
| 构建提示页面或链接不存在 | 核对导航路径、文件名大小写及相对链接 |
| 推送后没有触发构建 | 核对分支、变更路径、Actions 是否启用，以及工作流文件是否已经提交 |
| Pages 部署失败 | 检查 Pages Source 是否为 GitHub Actions，及 `github-pages` 环境保护规则 |
| 打开网站仍是旧版本 | 查看默认分支最新工作流是否成功，确认没有旧流程覆盖部署 |
| 公式或流程图未显示 | 检查浏览器是否允许访问相应 CDN，以及网络或扩展是否拦截脚本 |

参考：[MkDocs 配置](https://www.mkdocs.org/user-guide/configuration/)、[GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。
