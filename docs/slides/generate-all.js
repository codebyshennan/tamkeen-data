import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function getSubmodules(moduleDir) {
    const items = fs.readdirSync(moduleDir);
    return items.filter(item => {
        const fullPath = path.join(moduleDir, item);
        return fs.statSync(fullPath).isDirectory() &&
            !item.startsWith('_') &&
            !item.startsWith('.');
    });
}

function generateSlidesForSubmodule(submodulePath) {
    const slidesDir = path.join(submodulePath, 'slides');
    if (!fs.existsSync(slidesDir)) {
        fs.mkdirSync(slidesDir, { recursive: true });
    }

    // Read the README.md to get the title and content
    const readmePath = path.join(submodulePath, 'README.md');
    if (!fs.existsSync(readmePath)) {
        console.log(`No README.md found in ${submodulePath}`);
        return;
    }

    const readmeContent = fs.readFileSync(readmePath, 'utf8');
    const title = readmeContent.split('\n')[0].replace('# ', '');
    const subtitle = readmeContent.split('\n')[1]?.replace('## ', '') || '';

    // Get all markdown files in the submodule
    const files = fs.readdirSync(submodulePath)
        .filter(file => file.endsWith('.md') && file !== 'README.md')
        .map(file => ({
            name: file.replace('.md', ''),
            content: fs.readFileSync(path.join(submodulePath, file), 'utf8')
        }));

    // Generate slides data
    const slides = [
        {
            type: 'title',
            title: title,
            subtitle: subtitle
        }
    ];

    // Add content slides for each markdown file
    files.forEach(file => {
        const content = file.content;
        const lines = content.split('\n');
        const items = lines
            .filter(line => line.startsWith('- '))
            .map(line => line.replace('- ', ''));

        if (items.length > 0) {
            slides.push({
                type: 'content',
                title: file.name.split('-').map(word =>
                    word.charAt(0).toUpperCase() + word.slice(1)
                ).join(' '),
                content: {
                    type: 'box',
                    items: items
                }
            });
        }
    });

    // Write data.json
    const data = {
        title: title,
        subtitle: subtitle,
        slides: slides
    };

    fs.writeFileSync(
        path.join(slidesDir, 'data.json'),
        JSON.stringify(data, null, 4)
    );

    // Build the slides
    execSync(`node ${path.join(__dirname, 'build.js')} ${submodulePath}`);
}

function generateAllSlides() {
    const docsDir = path.join(__dirname, '..');
    const modules = fs.readdirSync(docsDir)
        .filter(item => {
            const fullPath = path.join(docsDir, item);
            return fs.statSync(fullPath).isDirectory() &&
                !item.startsWith('_') &&
                !item.startsWith('.') &&
                item !== 'slides' &&
                item !== 'assets';
        });

    modules.forEach(module => {
        const modulePath = path.join(docsDir, module);
        const submodules = getSubmodules(modulePath);

        submodules.forEach(submodule => {
            const submodulePath = path.join(modulePath, submodule);
            console.log(`Generating slides for ${submodulePath}`);
            generateSlidesForSubmodule(submodulePath);
        });
    });
}

generateAllSlides(); 
