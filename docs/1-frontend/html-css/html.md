# HTML

## Learning Objectives

1. Understand what HTML is and what it is used for
2. Understand basic HTML document structure
3. Understand how to use common tags

## Introduction

```html
<!DOCTYPE html>
<html>
  <head>
    <title>My First Page</title>
  </head>
  <body>
    <h1>My First Header</h1>
    <p>My first paragraph</p>
  </body>
</html>
```

HTML (HyperText Markup Language) defines elements on web pages. All web pages, even the most complex ones rely on HTML to represent their elements. In upcoming modules we will learn how to use CSS and JS to apply styling and interactivity to HTML elements.

HTML comprises tags and content between them. Web browsers read HTML and render content between tags based on tag specifications. For example, browsers will render content between "Header 1" (`h1`) opening (`<h1>`) and closing (`</h1>`) tags as large headers, and content between opening and closing "Paragraph" tags (`p`) in paragraph format.

```html
<h1>My First Heading</h1>
<p>My first paragraph.</p>
```

## Basic HTML Structure

All HTML documents generally start with the following declaration to use the latest version of HTML.

```html
<!DOCTYPE html>
```

After the `DOCTYPE` declaration is typically a set of `html` opening and closing tags surrounding all page content.

```html
<!DOCTYPE html>
<html>
  Page content
</html>
```

The first set of tags within the outermost `html` tags is usually the `head` tags. `head` tags contain important site metadata such as title (what's displayed in the browser tab bar), SEO metadata and links to stylesheets for styling and JavaScript for interactivity.

```html
<!DOCTYPE html>
<html>
  <head>
    <title>My First Page</title>
  </head>
</html>
```

`body` tags typically follow `head` tags. `body` tags contain the content of the page. The following example includes `body` tags with `h1` and `p` tags within them, specifying content to render on the page.

```html
<!DOCTYPE html>
<html>
  <head>
    <title>My First Page</title>
  </head>
  <body>
    <h1>My First Header</h1>
    <p>My first paragraph</p>
  </body>
</html>
```

That is the basic structure of all HTML pages. Feel free to play around with live examples on <a href="https://www.w3schools.com/html/html_examples.asp" target="_blank">W3Schools</a>.

## Common HTML Tags

### Summary

The following are common HTML tags we are most likely to use and encounter. Block elements occupy full page width and inline elements only occupy width of their content.

| Tag name              | Description                                             | Block vs inline |
| --------------------- | ------------------------------------------------------- | --------------- |
| `div`                 | Divider tag. Serves as group for other tags.            | Block           |
| `span`                | Span tag. Apply styles to inline content.               | Inline          |
| `h1`, `h2`, ..., `h6` | Header tags. `h1` is largest and `h6` is smallest.      | Block           |
| `p`                   | Paragraph tag. Used to separate paragraphs of text.     | Block           |
| `strong`, `em`        | Bold and italicise tags.                                | Inline          |
| `a`                   | Anchor tag. Link to another page with a URL.            | Inline          |
| `img`                 | Image tag. Render an image.                             | Inline          |
| `ol`, `ul`, `li`      | Ordered list, unordered list, list item. Render a list. | Block           |
| `table`, `tr`, `td`   | Table, table row, table data. Render a table.           | Block           |

### Anchor Tags (`a`)

Anchor tags link to other webpages and require an `href` parameter that contains a URL.

```html
<a href="google.com">Google</a>
```

To make the link open in a new tab, include the parameter `target="_blank"`.

```html
<a href="google.com" target="_blank">Google</a>
```

### Image Tags (`img`)

Image tags are self-closing and do not have separate opening and closing tags. They require `src` and `alt` parameters representing the source of the image and alternate text describing the image for accessibility, SEO and to display if the image is not available. `src` can be either a file path or a URL.

```html
<img src="images/googlelogo.png" alt="Google logo!" />
```

In this scenarion, the `src` of the image is a path to a folder called `images` and it looks for the file `googlelogo.png` in the said folder.

We can also wrap tags in each other to combine their functionality. For example, we can make an image a link by wrapping an `img` tag with an `a` tag.

```html
<a href="google.com" target="_blank">
  <img src="images/googlelogo.png" alt="Google logo!" />
</a>
```

### List Tags (`ol`, `ul`, `li`)

Wrap lists with `ol` (ordered) or `ul` (unordered) and wrap each list item with `li`.


```html
<ol>
  <li>Study</li>
  <li>Practise</li>
  <li>Success</li>
</ol>
```


```html
<ul>
  <li>Great students</li>
  <li>Great teachers</li>
  <li>Great school</li>
</ul>
```


### Table Tags (`table`, `tr`, `th`, `td`)

Wrap tables with `table`, table rows with `tr`, table headers in the 1st row with `th` and table data in subsequent rows with `td`.

```html
<table>
  <tr>
    <th>Company</th>
    <th>Contact</th>
    <th>Country</th>
  </tr>
  <tr>
    <td>Alfreds Futterkiste</td>
    <td>Maria Anders</td>
    <td>Germany</td>
  </tr>
  <tr>
    <td>Centro comercial Moctezuma</td>
    <td>Francisco Chang</td>
    <td>Mexico</td>
  </tr>
</table>
```

### Basic HTML Document


```html
<pre class="language-html"><code class="lang-html"><!DOCTYPE html>
<html>
  <head>
    <title>Coding Bootcamp</title>
  </head>
  <body>
    <h1>Welcome to Coding Bootcamp!</h1>
    <a href="google.com" target="_blank">
      <img src="images/googlelogo.png" alt="Google Logo!" />
    </a>
    <ol>
      <li>Study</li>
      <li>Practise</li>
      <li>Success</li>
    </ol>
    <h2>What are we going to learn? </h2>
    <table>
      <tr>
        <th>Topic</th>
        <th>Module</th>
        <th>Difficulty</th>
      </tr>
      <tr>
        <td>React</td>
        <td>One</td>
        <td>Easy</td>
      </tr>
      <tr>
        <td>Firebase</td>
        <td>Two</td>
        <td>Intermediate</td>
<strong>      </tr>
</strong>      <tr>
        <td>ExpressJs</td>
        <td>Three</td>
        <td>Advanced</td>
      </tr>
    </table>
  </body>
</html>
</code></pre>
```

Notice how this html document is composed of a single `html` tag, which contains the `head` and `body` tags. Then as stated above, the head tag contains the required meta data for SEO (Search Engine Optimization), but it also provides the browser with key information concerning what it should display. We have used a title tag such that the tab in the browser would read 'Coding Bootcamp'. With this in mind the `head` allows us to insert stylesheets which would inform the browser of how to display and render out html content. In the next section will cover how you can style and how you amend the `head` tag to link required CSS. 

Note that there is also only one body tag, this is where all of the html markup should be placed. Think of html as the structure of your website.  Developers use html elements to render information onto the browser, choosing the specific tag for the information type.

### Semantic HTML

[Semantic HTML tags](https://www.w3schools.com/html/html5_semantic_elements.asp) are part of a feature in HTML5 that provides context and meaning to both the browser and the developer.

Many web sites have divs with ids like: `<div id="nav">` `<div class="header">` and `<div id="footer">` to indicate different parts of a page such as navigation, header, and footer.

Using semantic elements, we can define different parts of a web page, here are some of them: 
- `<article>`
- `<aside>`
- `<footer>`
- `<header>`
- `<nav>`

The use of semantic tags help developers identify different parts of the page but are optional.