# Overleaf Setup Instructions

## Quick Start

This repository is ready to be imported into Overleaf for compilation and publication preparation.

## Files Required for Overleaf

The following files are necessary and included:

1. **channel_capacity_prl.tex** - Main manuscript file
2. **black_hole_capacity.png** - Figure showing channel capacity near black hole horizon

## Importing to Overleaf

### Method 1: GitHub Integration (Recommended)

1. Log in to your Overleaf account at https://www.overleaf.com
2. Click "New Project" → "Import from GitHub"
3. Select this repository: `SajithJude/QG-research`
4. Overleaf will automatically sync all files
5. Set the main document to: `channel_capacity_prl.tex`
6. Set the compiler to: `pdfLaTeX` (or `LaTeX`)

### Method 2: Direct Upload

1. Log in to Overleaf
2. Click "New Project" → "Upload Project"
3. Create a ZIP file containing:
   - channel_capacity_prl.tex
   - black_hole_capacity.png
4. Upload the ZIP file
5. Overleaf will extract and set up the project

## Compilation Settings

- **Compiler**: pdfLaTeX (default)
- **Main document**: channel_capacity_prl.tex
- **TeX Live version**: 2022 or later (for best RevTeX4-2 support)

## Expected Output

The manuscript should compile without errors into a 4-page Physical Review Letters format document with:

- Title: "Distance from Channel Capacity in Quantum Causal Networks"
- Author: Jittendran Jude Sajith Hector
- Abstract with key results
- Main content in two-column format
- One figure (black hole capacity profile)
- Embedded bibliography with 8 references
- PACS numbers for classification

## Compilation Process

Overleaf will automatically:
1. Compile the .tex file using pdfLaTeX
2. Include the figure from black_hole_capacity.png
3. Format the bibliography
4. Generate the final PDF

You can also compile manually using:
```bash
pdflatex channel_capacity_prl.tex
pdflatex channel_capacity_prl.tex
```
(Run twice to resolve all references)

## Troubleshooting

### If compilation fails:

1. **Check compiler**: Ensure pdfLaTeX is selected
2. **Check TeX Live version**: RevTeX4-2 requires TeX Live 2020+
3. **Check logs**: Look for error messages in the compile log
4. **Verify files**: Ensure both .tex and .png files are present

### Common Issues:

- **"File not found" for figure**: Ensure `black_hole_capacity.png` is in the same directory as the .tex file
- **RevTeX errors**: Update to latest TeX Live version in project settings
- **Bibliography errors**: The bibliography is embedded, so no .bib file is needed

## Post-Compilation

After successful compilation:

1. **Download PDF**: Click "Download PDF" to save locally
2. **Share link**: Use Overleaf's sharing feature for collaborators
3. **Version control**: Overleaf provides automatic version history
4. **Track changes**: Enable track changes for editorial revisions

## Submission Preparation

Before journal submission:

1. ✓ Verify author information is correct
2. ✓ Check all figures render properly
3. ✓ Review all equations for formatting
4. ✓ Verify all references are cited correctly
5. ✓ Check page count (should be ~4 pages for PRL)
6. ✓ Review acknowledgments
7. ✓ Generate final PDF from Overleaf

## Additional Notes

- The manuscript uses **RevTeX4-2**, which is the standard for Physical Review journals
- The embedded bibliography follows Physical Review format
- The document is set to two-column format per PRL guidelines
- PACS numbers are included for proper classification
- All equations are numbered and properly formatted

## Support

If you encounter any issues:
- Overleaf documentation: https://www.overleaf.com/learn
- RevTeX documentation: https://journals.aps.org/revtex
- Contact Overleaf support through their help center

---

**Status**: ✅ Ready for Overleaf import and compilation
**Last Updated**: October 29, 2025
