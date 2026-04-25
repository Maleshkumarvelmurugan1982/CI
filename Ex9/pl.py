from math import gcd

# Take input
WA = int(input("Enter white balls in Bag A: "))
BA = int(input("Enter black balls in Bag A: "))
WB = int(input("Enter white balls in Bag B: "))
BB = int(input("Enter black balls in Bag B: "))

print("\nINPUT VALUES")
print(f"Bag A: {WA} white, {BA} black")
print(f"Bag B: {WB} white, {BB} black")
print()

# Step 1: totals
total_A = WA + BA
total_B_after = WB + BB + 1

print("STEP 1: Total balls")
print(f"Total in A = {total_A}")
print(f"Total in B after transfer = {total_B_after}")
print()

# Case 1: pick white
print("CASE 1: Pick WHITE from A")
print(f"P(White from A) = {WA}/{total_A}")

print("After adding white to B:")
print(f"White in B = {WB + 1}")
print(f"Black in B = {BB}")

print(f"P(Black from B | White moved) = {BB}/{total_B_after}")

num1 = WA * BB
den = total_A * total_B_after

print(f"Contribution = ({WA}/{total_A}) * ({BB}/{total_B_after})")
print(f"= {num1}/{den}")
print()

# Case 2: pick black
print("CASE 2: Pick BLACK from A")
print(f"P(Black from A) = {BA}/{total_A}")

print("After adding black to B:")
print(f"White in B = {WB}")
print(f"Black in B = {BB + 1}")

print(f"P(Black from B | Black moved) = {BB + 1}/{total_B_after}")

num2 = BA * (BB + 1)

print(f"Contribution = ({BA}/{total_A}) * ({BB + 1}/{total_B_after})")
print(f"= {num2}/{den}")
print()

# Final addition
print("STEP 3: Add both cases")
numerator = num1 + num2
denominator = den

print(f"Total = {num1}/{den} + {num2}/{den}")
print(f"= {numerator}/{denominator}")
print()

# Simplify
print("STEP 4: Simplify using GCD")
g = gcd(numerator, denominator)
print(f"GCD({numerator}, {denominator}) = {g}")

numerator //= g
denominator //= g

sivajothi@sivajothi-HP-Laptop-15s-fq5xxx:~/Desktop$ 
