import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function generateSlideHtml(slide) {
    // Common elements for all slides
    const footerHtml = `<div class="slide-footer">Data Science & Analytics | Skills Union</div>`;
    const slideNumberHtml = `<div class="slide-number"></div>`;
    switch (slide.type) {
        case 'title': {
            return `
                <section>
                    <div class="content-wrapper">
                        <h1>${slide.title}</h1>
                        <h3>${slide.subtitle}</h3>
                    </div>
                    ${footerHtml}
                    ${slideNumberHtml}
                </section>
            `;
        }
        case 'content': {
            const content = slide.content;
            let itemsHtml = '';

            if (content.items) {
                const listType = content.ordered ? 'ol' : 'ul';
                itemsHtml = `
                    <div class="list-container">
                        <${listType}>
                            ${content.items.map(item => `<li>${item}</li>`).join('\n')}
                        </${listType}>
                    </div>
                `;
            }

            return `
                <section>
                    <div class="content-wrapper">
                        <h2>${slide.title}</h2>
                        <div class="box">
                            ${content.title ? `<h3>${content.title}</h3>` : ''}
                            ${itemsHtml}
                        </div>
                    </div>
                    ${footerHtml}
                    ${slideNumberHtml}
                </section>
            `;
        }
        default:
            return '';
    }
}

function buildSlides(moduleDir) {
    // Construct paths
    const templatePath = path.join(__dirname, 'template.html');
    const dataPath = path.join(moduleDir, 'slides', 'data.json');
    const outputPath = path.join(moduleDir, 'slides', 'index.html');

    // Ensure the slides directory exists
    const slidesDir = path.dirname(outputPath);
    if (!fs.existsSync(slidesDir)) {
        fs.mkdirSync(slidesDir, { recursive: true });
    }

    // Read the template and data files
    const template = fs.readFileSync(templatePath, 'utf8');
    const data = JSON.parse(fs.readFileSync(dataPath, 'utf8'));

    // Generate HTML for all slides
    const slidesHtml = data.slides.map(generateSlideHtml).join('\n');

    // Replace placeholders
    const output = template
        .replace('{{title}}', data.title)
        .replace('{{content}}', slidesHtml);

    // Write the output file
    fs.writeFileSync(outputPath, output);
    console.log(`Slides built successfully: ${outputPath}`);
}

// Get the module directory from command line argument
const moduleDir = process.argv[2];
if (!moduleDir) {
    console.error('Please provide the module directory as an argument');
    process.exit(1);
}

buildSlides(path.resolve(process.cwd(), moduleDir)); 
