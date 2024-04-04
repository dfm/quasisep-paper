## Technical details

This project follows a similar workflow to the one used by [showyourwork!](showyourwork), but it has
been slightly reimagined based on limitations of earlier iterations. Like showyourwork, this project
uses [Snakemake] to manage the workflow and track dependencies, but instead of Snakemake's built in
conda support, this project uses [pixi](pixi) for more explicit control over dependencies. Another
change compared to showyourwork, is that here we use the [MyST](myst) ecosystem of technical
authoring tools to allow the manuscript to be written in (MyST) Markdown, and converted to TeX for
publication.

To generate the paper, install [pixi](pixi) (you shouldn't need anything else) and run:

```bash
pixi run build
```

This will install the appropriate versions of all the dependencies (as specified in the `pixi.lock`
file), and then run the code to generate all the figures, and compile the paper. The output will be
saved in the `_build` directory.

[showyourwork]: https://github.com/showyourwork/showyourwork
[snakemake]: https://snakemake.readthedocs.io
[pixi]: https://pixi.sh
[myst]: https://mystmd.org