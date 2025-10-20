import datetime

def days_until_date_iso(s: str) -> int:
    target = datetime.date.fromisoformat(s.strip())
    return (target - datetime.date.today()).days

if __name__ == "__main__":
    s = input("Enter target date (YYYY-MM-DD): ").strip()
    try:
        days = days_until_date_iso(s)
    except Exception:
        print("Invalid format — use YYYY-MM-DD")
    else:
        if days > 0:
            print(f"{days} day(s) until {s}")
        elif days == 0:
            print("That date is today.")
        else:
            print(f"{-days} day(s) since {s}")
