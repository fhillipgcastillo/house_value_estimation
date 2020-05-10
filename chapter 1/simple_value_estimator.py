#manual estimation home
def estimate_home(size_in_sqft, num_bedroom):
  #assume all homes are wroth at elast $50,000
  baseValue = 50000
  # square feet cost
  sqftCost = 92

  #ssize of the house * value of 1 sqft
  valueBySize = baseValue + (size_in_sqft * sqftCost)

  bedRoomIncValue = 10000
  estimation = valueBySize + (num_bedroom * bedRoomIncValue)
  return estimation


estimate = estimate_home(3800, 5)
print("Estimated value:")
print(estimate)