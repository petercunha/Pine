from lib import pine
import os

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

print('''
====================================
 Pine: Neural-Network Aimbot (v0.1)
====================================

[INFO] press '0' to quit or ctrl+C in console...''')

pine.start(ENABLE_AIMBOT=True)
