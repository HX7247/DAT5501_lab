days_in_a_month = int(input("Enter number of days in the month: "))
if days_in_a_month < 28 or days_in_a_month > 31:
    print("Invalid number of days. Please enter a value between 28 and 31.")
    exit()

start_day = input("Enter starting day of the week: ")
if start_day not in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
    print("Invalid day. Please enter a valid day of the week. In the format: Monday, Tuesday, etc.")
    exit()

for day in range(1, days_in_a_month + 1):
    print(f"{day:2}", end=" ")
    if (day + ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(start_day)) % 7 == 0:
        print()


