"""
Render the notebooks

For each notebook in the directory:

1. jupyter nbconvert --to markdown name.ipynb
2. Replace "name_files/" with "/assets/notebooks/name_files/""
3. mv name.md ../_includes/notebooks/
4. mv name_files ../assets/notebooks/

This should then be followed by a pull request to the repository with the
given changes. i.e. this script is only to make it easier to render everything
and put it in the right place.

"""

import sys
import os
import re
import shutil
import subprocess
import nbformat
from traitlets.config import Config
from nbconvert import MarkdownExporter


def render():
    path_notebooks = '_notebooks'

    # Clear the output directories
    path_markdown = '_includes/notebooks'
    path_toc = '_includes/notebooks/toc'
    path_output = 'assets/notebooks'

    shutil.rmtree(path_markdown)
    os.makedirs(path_markdown)
    os.makedirs(path_toc)
    shutil.rmtree(path_output)
    os.makedirs(path_output)

    # Create new output
    for fname in os.listdir(path_notebooks):
        fpath = os.path.join(path_notebooks, fname)

        # Ignore directories
        if not os.path.isfile(fpath):
            continue

        # Ignore non-notebooks
        name, ext = fname.split('.', 2)
        if not ext == 'ipynb':
            continue

        # Set the appropriate output location (note: this is relative
        # to the website root /)
        www_output_files_dir = (
            'assets/notebooks/%s_files' % name)
        path_output_files_dir = 'assets/notebooks/%s_files' % name

        # Render the notebook
        with open(fpath) as f:
            r = {'output_files_dir': www_output_files_dir}

            # Read the notebook
            nb = nbformat.read(f, as_version=4)
            md_exporter = MarkdownExporter()
            (body, resources) = md_exporter.from_notebook_node(nb, resources=r)

            # Handle output files
            if not os.path.exists(path_output_files_dir):
                os.makedirs(path_output_files_dir)

            for www_name, v in resources['outputs'].items():
                # Save the output file to the correct location
                with open(www_name, 'wb') as fw:
                    fw.write(v)

            # Replace output paths to get relative urls
            search = r'\b%s/(.*)\b' % www_output_files_dir
            replace = r'{{ "/%s/\1" | relative_url }}' % www_output_files_dir
            body = re.sub(search, replace, body)

            # Write markdown file
            outname = '%s.md' % name
            outpath = os.path.join(path_markdown, outname)
            with open(outpath, 'w') as fw:
                fw.write(body)

            # Construct the table of contents
            cmd = ("pandoc --template=_toc-template.md --toc -t markdown %s"
                   % outpath)
            ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
            output = ps.communicate()[0]
            tocpath = os.path.join(path_toc, outname)
            with open(tocpath, 'wb') as fw:
                fw.write(output)

if __name__ == '__main__':
    render()
