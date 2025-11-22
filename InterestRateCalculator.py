def CompoundInterestCalculator(savings, annualInterestRate, years):
    # Convert annual interest rate from percentage to decimal
    multiply = annualInterestRate / 100  

    # Apply compound interest year by year
    for i in range(years):
        savings *= 1 + multiply               # Update savings each year
        text1 = "Your savings after " + str(i + 1) + " years is " + str(savings)
        print(text1)                          # Print yearly savings

    # Rule of 72 estimate for doubling time
    number_of_years_to_double = 72 / annualInterestRate

    # Final summary message
    text = (
        "Your savings after " + str(years) + " years is " + str(savings) +
        ". Your savings will double in " + str(number_of_years_to_double) + " years."
    )
    return text

# Example usage
print(CompoundInterestCalculator(1000, 50, 10))
