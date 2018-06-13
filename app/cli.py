from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit import prompt

from query_handler import predict

print("Welcome to PreCrimeBot!")
print("You can ask me to predict crime anywhere in London. Let's get started!")
print()
print("Each prediction will require a date and address as input.")
print("Don't worry, I understand most date and address formats.")
print("I will try to tell you when I can't understand you.")
print()
print("PS: type 'exit' and hit ENTER at any time to exit, or use CTLR+D")

address_history = InMemoryHistory()
date_history = InMemoryHistory()

while True:
    print('-' * 50)

    address = prompt("Address: ", history=address_history)
    if 'exit' == address:
        break

    date = prompt("Date: ", history=date_history)
    if 'exit' == date:
        break

    try:
        result = predict(date, address)
    except ValueError as e:
        print("Woops! Error: '%s'" % str(e))
        continue

    print("Here are the crime predictions you asked for:")
    print(" - theft %.2f%%" % round(result.theft * 100, 2))
    print(" - serious crime: %.2f%%" % round(result.serious * 100, 2))
    print(" - minor and other crime: %.2f%%" % round(result.other * 100, 2))

