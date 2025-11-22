# Get number of days in month from user
days_in_a_month = int(input("Enter number of days in the month: "))
# Validate day count (28-31)
if days_in_a_month < 28 or days_in_a_month > 31:
    print("Invalid number of days. Please enter a value between 28 and 31.")
    exit()

# Get starting day of week from user
start_day = input("Enter starting day of the week: ")
# Validate day name
if start_day not in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
    print("Invalid day. Please enter a valid day of the week. In the format: Monday, Tuesday, etc.")
    exit()

# Define days of week list for reuse
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Print header row
print("Mon Tue Wed Thu Fri Sat Sun")

# Add leading spaces based on starting day
for day in range(days_of_week.index(start_day)):
    print("    ", end="")

# Print each day of the month
for day in range(1, days_in_a_month + 1):
    print(f"{day:2}", end="  ")
    # Start new line when reaching end of week (Sunday)
    if (day + days_of_week.index(start_day)) % 7 == 0:
        print()


