#!/usr/bin/env node

(function () {
    const isNode = typeof module !== 'undefined' && module.exports;

    async function loadBrowserComponents() {
        const sections = document.querySelectorAll('[data-component]');
        for (const section of sections) {
            const file = section.getAttribute('data-component');
            try {
                const response = await fetch(`components/${file}`);
                if (!response.ok) {
                    throw new Error(`${response.status} ${response.statusText}`);
                }
                section.innerHTML = await response.text();
            } catch (error) {
                section.innerHTML = `<p class="component-error">Unable to load ${file}: ${error.message}</p>`;
            }
        }

        if (window.MathJax && window.MathJax.typesetPromise) {
            await window.MathJax.typesetPromise();
        }
    }

    async function buildStaticSite() {
        const fs = require('fs');
        const path = require('path');
        const { HTMLToMarkdownConverter } = require('./html-to-markdown.js');
        const { createManuscriptContext, injectPlaceholders } = require('./manuscript-data.js');

        const siteDir = __dirname;
        const distDir = path.join(siteDir, 'dist');
        const manifest = JSON.parse(fs.readFileSync(path.join(siteDir, 'manifest.json'), 'utf8'));
        let index = fs.readFileSync(path.join(siteDir, 'index.html'), 'utf8');
        const context = createManuscriptContext();

        const components = manifest.sections
            .slice()
            .sort((a, b) => a.order - b.order)
            .map((section) => {
                const componentPath = path.join(siteDir, 'components', section.file);
                const html = fs.existsSync(componentPath)
                    ? injectPlaceholders(fs.readFileSync(componentPath, 'utf8'), context)
                    : `<p>Missing component: ${section.file}</p>`;
                return `<section id="${section.id}" class="manuscript-section" data-section="${section.title}">\n${html}\n</section>`;
            })
            .join('\n\n');

        // Use a callback so manuscript text is inserted literally.  Passing
        // `components` as a replacement string makes JavaScript interpret
        // every LaTeX/entity sequence such as `$&ndash;` as the special `$&`
        // token (the entire regex match), recursively injecting the component
        // template into numeric ranges and corrupting the built HTML.
        index = index.replace(
            /<section id="abstract"[\s\S]*?<section id="(?:references|appendix)" data-component="[^"]+"><\/section>/,
            () => components
        );
        index = index.replace('<script src="build.js"></script>', '<script>window.__STATIC_MANUSCRIPT__ = true;</script>');

        fs.rmSync(distDir, { recursive: true, force: true });
        fs.mkdirSync(distDir, { recursive: true });
        fs.writeFileSync(path.join(distDir, 'index.html'), index, 'utf8');
        fs.copyFileSync(path.join(siteDir, 'simple-styles.css'), path.join(distDir, 'simple-styles.css'));
        if (fs.existsSync(path.join(siteDir, 'styles.css'))) {
            fs.copyFileSync(path.join(siteDir, 'styles.css'), path.join(distDir, 'styles.css'));
        }
        fs.copyFileSync(path.join(siteDir, 'manifest.json'), path.join(distDir, 'manifest.json'));

        // Copy root-level static files
        for (const file of ['robots.txt', 'sitemap.xml']) {
            const src = path.join(siteDir, 'public', file);
            if (fs.existsSync(src)) {
                fs.copyFileSync(src, path.join(distDir, file));
            }
        }

        // Copy public directory to dist
        const publicDir = path.join(siteDir, 'public');
        const publicDest = path.join(distDir, 'public');
        if (fs.existsSync(publicDir)) {
            copyRecursiveSync(publicDir, publicDest, fs, path);
        }

        // Copy assets directory to dist
        const assetsDir = path.join(siteDir, 'assets');
        const assetsDest = path.join(distDir, 'assets');
        if (fs.existsSync(assetsDir)) {
            copyRecursiveSync(assetsDir, assetsDest, fs, path);
        }

        copyResults(path.join(siteDir, '..', 'results'), distDir, fs, path);

        // Copy results/figures to dist/figures (component image references use figures/)
        const resultsFiguresDir = path.join(siteDir, '..', 'results', 'figures');
        const distFiguresDir = path.join(distDir, 'figures');
        if (fs.existsSync(resultsFiguresDir)) {
            copyRecursiveSync(resultsFiguresDir, distFiguresDir, fs, path);
        }

        // Also copy site/figures to dist/figures if it exists (overrides)
        const siteFiguresDir = path.join(siteDir, 'figures');
        if (fs.existsSync(siteFiguresDir)) {
            copyRecursiveSync(siteFiguresDir, distFiguresDir, fs, path);
        }

        for (const file of ['CITATION.cff', 'CITATION.bib', 'README.md', 'LICENSE', 'VERSION.json']) {
            const src = path.join(siteDir, '..', file);
            if (fs.existsSync(src)) {
                fs.copyFileSync(src, path.join(distDir, file));
            }
        }

        const converter = new HTMLToMarkdownConverter();
        await converter.convertSiteToMarkdown();

        console.log(`Built TEP-PPP static manuscript: ${path.join(distDir, 'index.html')}`);

        // Phase A4: fail the build if a headline claim has drifted from its committed output.
        try {
            const { execFileSync } = require('child_process');
            execFileSync('python3', [path.join(siteDir, '..', 'scripts', 'check_manuscript_numbers.py')],
                { stdio: 'inherit' });
        } catch (e) {
            console.error('check_manuscript_numbers.py FAILED — manuscript numbers are out of sync with results/outputs/');
            process.exitCode = 1;
        }
    }

    function copyResults(resultsDir, distDir, fs, path) {
        if (!fs.existsSync(resultsDir)) {
            return;
        }
        const figuresSrc = path.join(resultsDir, 'figures');
        const figuresDest = path.join(distDir, 'results', 'figures');
        if (fs.existsSync(figuresSrc)) {
            fs.mkdirSync(figuresDest, { recursive: true });
            for (const file of fs.readdirSync(figuresSrc)) {
                if (file.endsWith('.png')) {
                    fs.copyFileSync(path.join(figuresSrc, file), path.join(figuresDest, file));
                }
            }
        }

        const resultsDest = path.join(distDir, 'results');
        fs.mkdirSync(resultsDest, { recursive: true });
        for (const file of fs.readdirSync(resultsDir)) {
            if (file.endsWith('.json') || file.endsWith('.csv')) {
                fs.copyFileSync(path.join(resultsDir, file), path.join(resultsDest, file));
            }
        }
    }

    function copyRecursiveSync(src, dest, fs, path) {
        const stat = fs.statSync(src);
        if (stat.isDirectory()) {
            if (!fs.existsSync(dest)) {
                fs.mkdirSync(dest, { recursive: true });
            }
            const entries = fs.readdirSync(src);
            for (const entry of entries) {
                copyRecursiveSync(path.join(src, entry), path.join(dest, entry), fs, path);
            }
        } else {
            const parentDir = path.dirname(dest);
            if (!fs.existsSync(parentDir)) {
                fs.mkdirSync(parentDir, { recursive: true });
            }
            fs.copyFileSync(src, dest);
        }
    }

    if (isNode) {
        module.exports = { buildStaticSite };
        if (require.main === module) {
            buildStaticSite().catch((error) => {
                console.error(error);
                process.exit(1);
            });
        }
    } else if (!window.__STATIC_MANUSCRIPT__) {
        document.addEventListener('DOMContentLoaded', loadBrowserComponents);
    }
})();
