# How this course works

**After reading this:** you know how to use your time before and during live sessions, what to bring to office hours, and where grading rules live (your syllabus, not this page).

This bootcamp uses a **flipped classroom** model: you absorb new ideas on your own time from the written lessons and videos, then use live sessions to **practice**, **debug**, and **ask questions**. That only works if you show up having already skimmed the material, otherwise class feels like a first lecture instead of a lab.

## Helpful video

What "flipped" means in practice: learn the content on your own schedule, then use class for practice and questions.

<iframe width="560" height="315" src="https://www.youtube.com/embed/qdKzSq_t8k8" title="The Flipped Classroom Model" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Flipped classroom in plain language

| Typical "traditional" course | This course |
|------------------------------|-------------|
| Instructor explains everything in the room; you do hard work alone later | You read or watch first; the room is for doing and fixing |
| One pace for everyone | You can re-read and pause videos until it clicks |
| Few chances to ask questions | Live time is reserved for questions and hands-on help |

**What you should do before each live session:** open the module's pages for that week, run through the headings, and try at least one short exercise or code cell so you know *where* you are stuck (environment, syntax, or concept). "I didn't have time to look" is harder to fix in an hour than "I tried this and got this error."

## A concrete example week

This is illustrative, your instructor may adjust pacing. Use it as a template for how to split your time.

| When | What to do |
|------|------------|
| **Early in the week** | Finish [environment setup](./README.md) if you have not already. Skim the next module's `README` and one lesson page. |
| **Mid-week** | Work through one lesson end-to-end at your own speed. Note every error message and the line that triggered it. |
| **Before live session** | Re-run your notebook or script from a clean kernel or fresh terminal so you know it still works. Write down 1-3 specific questions (e.g. "why does this join duplicate rows?" not "I don't get SQL"). |
| **Live session** | Bring those questions, share your screen if asked, and work the in-class exercises. |
| **After** | Finish the assignment draft, then use office hours for blockers, not for first exposure to the topic. |

## Before, during, and after class

**Before class**

- Read or watch the assigned materials so you recognize terms when they appear in live demos.
- Run the code yourself; copy-pasting without running leaves gaps that show up under time pressure.
- Keep a small log: date, what you tried, exact error text. That log is gold for office hours.

**During class**

- Expect demonstrations, short exercises, and time to work while teaching staff circulate. It is not a long lecture block.
- Ask early. Silence often means "everyone is stuck on the same install issue"-say it out loud.

**After class**

- Turn notes into a working example in your own repo or folder.
- If something still fails at home, compare your environment to the [setup guides](./README.md) (Python version, conda env active, correct interpreter in the editor).

## Office hours

Your instructor shares the **time**, **link**, and **rules** (drop-in vs sign-up sheet). In general:

- **Bring:** your error message or a minimal notebook that reproduces the problem, and say what you already tried.
- **Good use of time:** "Here is my traceback and the three lines I changed."
- **Harder to fix in a few minutes:** "Explain the whole of statistics from scratch"-book that as a study-plan conversation, not a single office hour.

If your cohort uses a forum or chat, use the same discipline: paste errors and code snippets, not only "it doesn't work."

## How to unblock yourself

When you hit a wall, work through these three steps before asking for help:

1. **Trace the error message.** Read it carefully, what could be causing it? Fix that, then address the next error. If there is no visible error, find where the problem is using `print` statements or logging to give yourself more clues.
2. **Google the error and context.** Search for the exact error message plus the technology name, e.g. `Python NameError: name is not defined`. Skim results and dig deeper into the most promising ones.
3. **Ask peers or your instructor, with context.** Share what the error is, what you think is causing it, and what you learnt from steps 1 and 2. Context lets others help you quickly. We use mainstream technologies and the problems you encounter will rarely be exotic.

### How to use Google effectively

Professional data scientists spend a significant portion of their time finding answers on Google. Learning to search well may be your most transferable skill from this course.

**Search formula:** `<error message> <technology name>`, for example, `NameError: name is not defined Python`.

A few habits that help:

- When you land on a result (Stack Overflow, docs, forum), scan for the relevant part quickly, do not read everything top-to-bottom.
- If the first search doesn't work, try different keyword combinations. It often takes several attempts.
- With experience you will develop a feel for when you are on the right track; trust that feeling and keep refining.

### How to ask questions that get answers

Always include context. A question without context forces the person helping you to guess, which wastes both of your time.

For technical questions, useful context includes:

- What is the exact error message?
- What do you think the problem is?
- What have you already tried from debugging and googling?
- What is the relevant code?

**Compare these three versions of the same question:**

> **No context**: "My code is not working. Please help!"

> **Incomplete context**: "My code is not working. I'm getting `NameError: name is not defined`. Please help!"

> **Full context**: "I'm getting `NameError: name 'clases' is not defined` on line 3. I'm trying to access a variable named `clases`. Googling suggests the variable isn't defined, but I'm not sure why. Relevant code is below, any suggestions?"

The third question is easy to answer in one reply. The first two require multiple back-and-forth exchanges before anyone can help.

## Grades and milestones

How you are graded, assignments, projects, attendance expectations, late policy, comes from **your instructor and syllabus**, not from this site. If anything here conflicts with what you were told in class, follow your syllabus.

## Summary

- **You** preview and practice on your schedule.
- **Live time** is for application, debugging, and targeted questions.
- **Setup** must be solid early; almost every mysterious bug later traces back to environment or paths, use the prep guides until your toolchain is boringly reliable.
