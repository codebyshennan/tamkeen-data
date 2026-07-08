# Module 1: Data Fundamentals - Answer Key

## Overview

**Audience:** Instructors and TAs. Pair this file with the student-facing [Module 1 assignment](/broken/pages/eGJif8kIKFByFcOaf4OZ).

**Primary use:** Verify correct options, reuse explanations in feedback, and spot questions that need revision.

> **Contributors:** Style and structure follow `docs/meta/DOCUMENTATION_GUIDELINES.md` (`meta/` is excluded from the Jekyll build).

## Part 1: Introduction to Data Analytics (20 points)

1.  What is the primary purpose of data collection?

    * a. To analyze data
    * **b. To gather raw material for analysis** ✓
    * c. To ensure data accuracy
    * d. To visualize data

    _Explanation: Data collection is the foundational step that provides the raw material needed for all subsequent analysis. Without proper data collection, there would be nothing to analyze, clean, or visualize._
2.  Which of the following is an example of first-party data?

    * a. Social media insights
    * b. Market research reports
    * **c. Customer interactions** ✓
    * d. Data aggregators

    _Explanation: First-party data is collected directly from your customers or users through their interactions with your business, making customer interactions a prime example of first-party data._
3.  What does GDPR stand for?

    * **a. General Data Protection Regulation** ✓
    * b. General Data Privacy Regulation
    * c. General Data Protection Rules
    * d. Global Data Protection Regulation

    _Explanation: GDPR (General Data Protection Regulation) is the European Union's comprehensive data protection law that came into effect in 2018._
4.  Which method is used for gathering qualitative data?

    * a. Surveys
    * b. Data logging
    * **c. Interviews** ✓
    * d. Web scraping

    _Explanation: Interviews are a primary method for gathering qualitative data as they allow for in-depth, open-ended responses and detailed exploration of subjects' experiences and perspectives._
5.  What is the main focus of predictive analytics?

    * a. Understanding past events
    * b. Analyzing current data
    * **c. Forecasting future outcomes** ✓
    * d. Summarizing data

    _Explanation: Predictive analytics uses historical and current data to make predictions about future events or behaviors, making forecasting future outcomes its primary focus._

## Part 2: Python Programming (20 points)

1.  Which of the following correctly declares a list in Python?

    * a. list = (1, 2, 3)
    * **b. list = \[1, 2, 3]** ✓
    * c. list = {1, 2, 3}
    * d. list = <1, 2, 3>

    _Explanation: In Python, square brackets \[] are used to declare a list. Parentheses () create a tuple, curly braces {} create a set or dictionary, and angle brackets <> are not used for collections._
2.  What is the output of `len(['a', 'b', ['c', 'd']])`?

    * a. 4
    * **b. 3** ✓
    * c. 2
    * d. Error

    _Explanation: The len() function counts the number of elements at the top level of the list. Here, there are three elements: 'a', 'b', and the nested list \['c', 'd']._
3.  Which statement correctly creates a function in Python?

    * a. function my\_func():
    * b. my\_func():
    * **c. def my\_func():** ✓
    * d. define my\_func():

    _Explanation: In Python, functions are defined using the 'def' keyword followed by the function name and parentheses._
4.  What is the purpose of the **init** method in a Python class?

    * a. To delete an object
    * **b. To initialize object attributes** ✓
    * c. To import modules
    * d. To create a loop

    _Explanation: The **init** method is a constructor that initializes the attributes of an object when it is created from a class._
5.  How do you import a specific function from a module?

    * a. import function from module
    * b. include module.function
    * **c. from module import function** ✓
    * d. using module.function

    _Explanation: In Python, you can import specific functions from a module using the 'from module import function' syntax._

## Part 3: Statistics (20 points)

1.  What is the purpose of probability in statistics?

    * a. To make data collection easier
    * b. To organize data
    * **c. To quantify uncertainty** ✓
    * d. To create graphs

    _Explanation: Probability is used in statistics to quantify and measure uncertainty in data and predictions, helping us understand the likelihood of different outcomes._
2.  In a normal distribution, where is the mean located?

    * a. At the leftmost point
    * b. At the rightmost point
    * **c. At the center** ✓
    * d. At both ends

    _Explanation: In a normal distribution, the mean is located at the center of the distribution, with the data being symmetrically distributed around it._
3.  What is the Monty Hall problem demonstrating?

    * a. Random sampling
    * b. Normal distribution
    * **c. Conditional probability** ✓
    * d. Standard deviation

    _Explanation: The Monty Hall problem is a famous example that demonstrates conditional probability and how our intuition about probability can sometimes be incorrect._
4.  What does a p-value of 0.05 indicate?

    * a. The null hypothesis is true
    * **b. There's a 5% chance of seeing these results by chance** ✓
    * c. The experiment failed
    * d. The data is normally distributed

    _Explanation: A p-value of 0.05 indicates that there is a 5% probability of observing results as extreme as the ones obtained if the null hypothesis is true._
5.  What is the purpose of a confidence interval?

    * a. To prove the hypothesis
    * b. To collect data
    * **c. To estimate parameter uncertainty** ✓
    * d. To calculate the mean

    _Explanation: Confidence intervals provide a range of plausible values for a population parameter, helping us understand the uncertainty in our estimates._

## Part 4: NumPy Operations (20 points)

1.  Which NumPy function creates an array filled with zeros?

    * **a. np.zeros()** ✓
    * b. np.empty()
    * c. np.null()
    * d. np.blank()

    _Explanation: np.zeros() is the correct function to create an array filled with zeros. np.empty() creates an uninitialized array, and np.null() and np.blank() don't exist in NumPy._
2.  Why is NumPy faster than regular Python lists?

    * a. It uses more memory
    * b. It has fewer features
    * **c. It uses vectorization** ✓
    * d. It's written in Python

    _Explanation: NumPy's speed comes from vectorization, which allows operations to be performed on entire arrays at once instead of using loops._
3.  What is broadcasting in NumPy?

    * a. Sending data over a network
    * **b. Automatic array shape matching** ✓
    * c. Converting arrays to lists
    * d. Printing array contents

    _Explanation: Broadcasting is NumPy's ability to perform operations between arrays of different shapes by automatically expanding the smaller array to match the larger one._
4.  Which operation creates a 2D array?

    * a. np.array(1,2,3)
    * **b. np.array(\[\[1,2], \[3,4]])** ✓
    * c. np.array(1:4)
    * d. np.2darray(\[1,2,3])

    _Explanation: A 2D array in NumPy is created using nested lists, where each inner list represents a row of the array._
5.  What is the purpose of np.arange()?

    * a. To sort an array
    * **b. To create an array with evenly spaced values** ✓
    * c. To reshape an array
    * d. To combine arrays

    _Explanation: np.arange() creates an array with evenly spaced values within a specified interval, similar to Python's range() but returning a NumPy array._

## Part 5: Pandas Data Analysis (20 points)

1.  Which method is used to handle missing values in a DataFrame?

    * a. df.remove()
    * **b. df.dropna()** ✓
    * c. df.delete()
    * d. df.clean()

    _Explanation: df.dropna() is the correct Pandas method to remove rows or columns containing missing values. The other options are not valid Pandas methods._
2.  What is the difference between a Pandas Series and DataFrame?

    * a. Series is faster than DataFrame
    * **b. Series is 1-dimensional, DataFrame is 2-dimensional** ✓
    * c. Series can only store numbers
    * d. DataFrame can only store text

    _Explanation: A Series is a one-dimensional labeled array that can hold data of any type, while a DataFrame is a two-dimensional labeled data structure with columns of potentially different types._
3.  How do you read a CSV file in Pandas?

    * a. pd.load\_csv()
    * b. pd.import\_csv()
    * **c. pd.read\_csv()** ✓
    * d. pd.open\_csv()

    _Explanation: pd.read\_csv() is the correct Pandas function to read data from a CSV file into a DataFrame._
4.  Which method shows the first few rows of a DataFrame?

    * a. df.first()
    * **b. df.head()** ✓
    * c. df.top()
    * d. df.start()

    _Explanation: df.head() is used to display the first few rows of a DataFrame, by default showing the first 5 rows._
5.  What does df.describe() do?

    * a. Shows column names
    * b. Lists data types
    * **c. Generates summary statistics** ✓
    * d. Counts rows

    _Explanation: df.describe() generates summary statistics of numerical columns in a DataFrame, including count, mean, std, min, 25%, 50%, 75%, and max._

## Grading Rubric

Each part is worth 20 points:

* 1 point per correct answer (20 questions per section)
* Total possible points: 100

Note: Partial credit may be given for questions that require explanation or show work process.
