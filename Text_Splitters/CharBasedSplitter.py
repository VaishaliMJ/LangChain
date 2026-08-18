"""----------------------------------------------------------------------------------
    Problem Statement   :   Length Based Text Splitters
    Author              :   Vaishali M. Jorwekar
----------------------------------------------------------------------------------"""
from langchain_text_splitters import CharacterTextSplitter



BORDER="-"*65



###############################################################################
#   Function        :   main
#   Input Params    :   None
#   Output Params   :   None
#   Description     :   Entry point of the program
#   Author          :   Vaishali M Jorwekar
###############################################################################
def main():
    text="""Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in engineering, mathematics, and computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximise their chances of achieving defined goals.[1]

    High-profile applications of AI include advanced web search engines, chatbots, virtual assistants, autonomous vehicles, play and analysis in strategy games (e.g., chess and Go), and content generation (e.g. images, audio, and videos).

    The traditional goals of AI research include learning, reasoning, knowledge representation, planning, natural language processing, and perception, as well as support for robotics.[a] To reach these goals, AI researchers use techniques including state space search and mathematical optimisation, formal logic, artificial neural networks, and methods based on statistics, operations research, and economics.[b] AI also draws upon psychology, linguistics, philosophy, neuroscience, and other fields.[2] Some companies, such as OpenAI, Google DeepMind, and Meta, aim to create artificial general intelligence (AGI)—AI that can complete nearly any cognitive task at least as well as a human.[3]

    """
    
    splitter=CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        separator=''
    )
    
    chunks=splitter.split_text(text)
    for item in range(len(chunks)):
        
        print(f"Chunk No {item} :    \n{chunks[item]}")
        print(BORDER)
###############################################################################
#   Entry point of the program
###############################################################################
if __name__=="__main__":
    main()

