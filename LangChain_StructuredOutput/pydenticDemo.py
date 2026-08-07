"""----------------------------------------------------------------------------------
    Problem Statement   :   Pydentic demo
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from pydantic import BaseModel,EmailStr,Field
from typing import Optional
BORDER="-"*60

class Student(BaseModel):
    name    :   str = "Siya"
    age     :   Optional[int] = None
    email : EmailStr
    cgpa    :   float=Field(gt=0,lt=10)
    
newStudent  =  {"age"  :   '22' , 
                "email"   :   "abc@gamil.com",
                "cgpa"  :   9}

student=Student(**newStudent)
print(f"Student Details  :   {student}")    

print(BORDER)
studentDict=dict(student)
print(f"Dictionary Format   :\n   {studentDict}")
print(BORDER)

print(BORDER)
studentJSONFormat=student.model_dump_json()
print(f"Json Format   :\n   {studentJSONFormat}")
print(BORDER)