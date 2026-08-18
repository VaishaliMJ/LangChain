"""----------------------------------------------------------------------------------
    Problem Statement   :   Recursive Character Text Splitter
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_text_splitters import RecursiveCharacterTextSplitter,Language




BORDER="-"*65



###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    text="""class Student:
    
    def __init__(self, name: str, student_id: int):
        self.name = name          # Instance variable for student name
        self.id = student_id      # Instance variable for student ID
        self.grades = []          # List to store numerical grades
        
    def add_grade(self, grade: float):
        if 0 <= grade <= 100:
            self.grades.append(grade)
        else:
            raise ValueError("Grade must be between 0 and 100.")
            
    def get_average(self) -> float:
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)
        
    def get_status(self) -> str:
        return "Pass" if self.get_average() >= 60 else "Fail"

    def __str__(self) -> str:
        return f"Student: {self.name} (ID: {self.id}) | Avg: {self.get_average():.2f}"
    """
    
    splitter=RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=100,
        chunk_overlap=0,
        
    )
    
    chunks=splitter.split_text(text)
    print(f"Number Of Chunks  : {len(chunks)}")
    print(BORDER)
    print(BORDER)
    for item in range(len(chunks)):
        
        print(f"Chunk No {item} :    \n{chunks[item]}")
        print(BORDER)
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()

