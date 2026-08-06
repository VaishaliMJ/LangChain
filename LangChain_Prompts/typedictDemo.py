"""----------------------------------------------------------------------------------
    Problem Statement   :   typedict demo
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from typing import TypedDict


class Person(TypedDict):
    name    :   str
    age :   int
    
newPerson   :   Person =    {'name' : "xyz",
                             "age"  :   22}

print(f"Person Details  :   {newPerson}")    