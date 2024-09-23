
# Operators
Operators are symbols that perform operations on variables and values.

The basic types of operators:



Arithmetic
Unary
Assignment
Relational
Logical / Conditional

# You can create a LearnOperators.java file and code along as we go through these operators.

public class LearnOperators {
  public static void main(String[] args) {

    int a = 10;
    int b = 20;
  }

}
# Arithmetic Operators:

### Arithmetic operator table to be added here



# Notes:

 + is also used to concatenate strings e.g. "Hello" + "World"
  becomes "HelloWorld".

System.out.println("a + b = " + (a + b));
System.out.println("a - b = " + (a - b));
System.out.println("a * b = " + (a * b));
System.out.println("a / b = " + (a / b));
System.out.println("b % a = " + (b % a));

# Unary Operator

### Unary Operator table to be added here

int unaryPlus = +10;
int unaryMinus = -10;
System.out.println("unaryPlus: " + unaryPlus);
System.out.println("unaryMinus: " + unaryMinus);

int preIncrement = ++unaryPlus;
System.out.println("preIncrement: " + preIncrement);

int preDecrement = --unaryMinus;
System.out.println("preDecrement: " + preDecrement);

int postIncrement = unaryPlus++;
System.out.println("postIncrement: " + postIncrement);

int postDecrement = unaryMinus--;
System.out.println("postDecrement: " + postDecrement);

boolean isTrue = true;
System.out.println("isTrue: " + isTrue);
System.out.println("!isTrue: " + !isTrue);

Observe the output of the above code. What is the difference between pre and post increment/decrement?

# Pre vs Post Increment/Decrement:

Pre-increment/decrement operators increment/decrement the value of the variable before returning the value.

Post-increment/decrement operators increment/decrement the value of the variable after returning the value.

int x = 10;
int y = 10;

System.out.println("x: " + x);
System.out.println("y: " + y);

System.out.println("x++: " + x++);
System.out.println("++y: " + ++y);

System.out.println("x: " + x);
System.out.println("y: " + y);

