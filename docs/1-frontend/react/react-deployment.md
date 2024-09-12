# React Deployment

## Learning Objectives

1. Deploy a ReactJS using Github Pages
2. Write yaml files that would contain jobs for deployment

## Deployment

Deployment is the process of making an application available to the public. This is typically done by hosting the application in a platform. 

Github has its own deployment platform which is called Github Pages. The application is hosted directly from a GitHub repository and the developer can edit, then push, and any changes would be live.

Deployment to Github pages can be done in the following ways:
1. via gh-pages CLI tool
2. via git push

### Vitejs Deployment: Github Pages using the gh-pages CLI tool

Navigate into your project via a CLI tool (ubuntu terminal / powershell)
Go through gitflow and save your current code.

```
git add
git commit -m 'commit-message'
git push 
```

Next, run the following command:

`npm run build`

You can check to see if your build worked by running the command:

`npm run preview`

You should be able to see your application within your browser at <a href="http://localhost:4173/" target="_blank">`http://localhost:4173`</a>

After validating that this works, we will install the required packages:


Run the command: `npm install gh-pages --save-dev`


Now we will setup the package.json:


Add these scripts into scripts:

```json
"predeploy": "npm run build",
"deploy": "gh-pages -d dist",
```

Now we will configure the `vite.config.js` file:
We need to add a key-value pair for gh-pages:

The key will be base, the value should be 'name-of-your-repo'
The configuration should look something like:

```js
export default defineConfig({
  plugins: [react()],
  base: "/sample_vitejs/",
});
```

Now you should be able to run the command: `npm run deploy`


### Vitejs Deployment: Github Pages through git push

Navigate into your project via a CLI tool (ubuntu terminal / powershell)

Go through gitflow and save your current code.

```sh
git add
git commit -m 'commit-message'
git push 
```

Run the following command:

`git checkout -b gh-pages`

Next, run the following command:

`npm run build`

You can check to see if your build worked by running the command:

`npm run preview`


You should be able to see your application within your browser at <a href="http://localhost:4173/" target="_blank">`http://localhost:4173`</a>

Now we will configure the `vite.config.js` file:
We need to add a key value pair for deployment:

The key will be base, the value should be 'name-of-your-repo'
The configuration should look something like:

```js
export default defineConfig({
  plugins: [react()],
  base: "/sample_vitejs/",
});
```

Within your local machine we need to make a new github workflow within a yml file. Here are the steps:

1. Create a new folder named `.github`
2. Within the newly created directory create a `workflows` folder
3. In the `workflows` folder create a new file named: `jekyll-gh-pages.yml`

4. Paste in this file:

```yml
# Simple workflow for deploying static content to GitHub Pages
name: Deploy static content to Pages

on:
  # Runs on pushes targeting the default branch
  push:
    branches: ["gh-pages"]

  # Allows you to run this workflow manually from the Actions tab
  workflow_dispatch:

# Sets the GITHUB_TOKEN permissions to allow deployment to GitHub Pages
permissions:
  contents: read
  pages: write
  id-token: write

# Allow one concurrent deployment
concurrency:
  group: "pages"
  cancel-in-progress: true

jobs:
  # Single deploy job since we're just deploying
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Setup Node.js environment
        uses: actions/setup-node@v4.0.0
        with:
          node-version: lts/*
          cache: 'npm'
      - name: Install dependencies
        run: npm install
      - name: Build
        run: npm run build
      - name: Setup Pages
        uses: actions/configure-pages@v3
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v2
        with:
          # Upload entire repository
          path: './dist'
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v2
```

5. In your GitHub repository, go to `Settings -> Pages` and then choose source as `GitHub Actions`, not deploy from branch. 

6. Now you should be able to deploy when you push to this branch, before this will work we need to go through git flow (add commit and push)
Then run the command:

`git push origin gh-pages`

*Note: make sure you are working on the gh-pages branch when pushing changes to github for the job to run.*