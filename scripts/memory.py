class ConversationMemory:
    """Holds the running conversation as a list of turns and manages prompt formatting."""

    def __init__(self,max_turns=10):
        # Initializes an empty list to hold our conversation dictionaries, and sets a maximum number of turns to retain in memory.
        self.turns = []
        self.max_turns = max_turns

    def add_turn(self, question, answer):
        # Adds a new turn to the conversation memory. Each turn is stored as a dictionary with 'question' and 'answer' keys.
        self.turns.append({"question": question, "answer":answer})

        # Keep only the most recent 'max_turns' to control prompt size/cost
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def as_text(self):
        # Converts the conversation history into a single string formatted for prompting. Each turn is separated by newlines, and questions/answers are labeled.
        
        if not self.turns:
            return "(no conversation history)"
        
        lines = []
        for turn in self.turns:
            lines.append(f"User : {turn['question']}")
            lines.append(f"AI: {turn['answer']}")

        return "\n".join(lines)
    
    def recent_questions(self, n=3):
        # Retrieves the most recent 'n' questions from the conversation history. This can be useful for providing context to the model without including full answers.
        questions = []
        for turn in self.turns[-n:]:
            questions.append(turn["question"])
        return questions
    
if __name__ == "__main__":
     # Create an instance of our memory class
    memory = ConversationMemory()

        # Simulate a conversation by adding some turns
    memory.add_turn("What is the capital of France?", "The capital of FRance is Paris.")
    memory.add_turn("What is the population of Paris?", "The population of Paris is around 2.1 million.")

    # Print the full conversation history as text
    print("Full Conversation History:")
    print(memory.as_text())
    # Print just the recent questions
    print("\nRecent Questions:")
    print(memory.recent_questions())

     


