import os
from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from openai import AzureOpenAI


load_dotenv(override=True)

class RAG:
    def __init__(self, retriever, difficulty="Medium", max_history_turns=5):
        # LLM Provider: 'ollama' or 'azure'
        self.llm_provider = os.getenv('LLM_PROVIDER', 'ollama').lower()

        # Model configuration based on provider
        if self.llm_provider == 'azure':
            self.llm_name = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-5')
        else:  # ollama
            self.llm_name = os.getenv('OLLAMA_MODEL', 'llama3.2')

        self.llm = self._setup_llm()
        self.retriever = retriever
        self.difficulty = difficulty

        # Persistent conversation history with configurable memory depth
        # Each turn = 1 user message + 1 assistant message
        self.conversation_history = []
        self.max_history_turns = max_history_turns  # Maximum number of Q&A pairs to retain

        # Memorizziamo separatamente l'ultima domanda dell'assistente
        self.last_question = None

        self.difficulty_instructions = {
            "Easy": """
                - Focus on basic concepts, definitions, and recall
                - Ask straightforward questions that test understanding of fundamental principles
                - Require simple explanations or direct applications
                - Suitable for beginners or introductory level
            """,
            "Medium": """
                - Focus on application and intermediate complexity
                - Ask questions that require understanding and applying concepts
                - May involve multi-step reasoning or combining multiple ideas
                - Suitable for intermediate university level
            """,
            "Hard": """
                - Focus on complex analysis, synthesis, and advanced reasoning
                - Ask questions requiring deep understanding and critical thinking
                - Involve multi-step problem solving, edge cases, or theoretical implications
                - Suitable for advanced university level or expert understanding
            """
        }

        self.qa_prompt_tmpl_str = """
            Context information is below.
                <context/user_query>:
                ---------------------
                {context}
                ---------------------

                Based on the context information above, generate **one single open-ended question** related to the query below. Follow these rules precisely:

                1. **Number of questions:** 1
                2. **Question type:** Open-ended (short essay or problem-solving)
                3. **Difficulty level:** {difficulty}
                {difficulty_instructions}
                4. **LaTeX formatting:**
                - Use inline math between single dollar signs `$...$`
                - Use block math between double dollar signs `$$...$$`
                - Do not escape backslashes
                5. **Tone:** Professional and educational
                6. **Output format:**

                    ```
                    **Open-ended Question**

                    [your question here]
                    ```

                7. **Interaction flow:**
                - Step 1: Generate and display the question only.
                - Step 2: Wait for the user to provide their answer.
                - Step 3: After receiving the user's answer, evaluate it:
                    - Positive feedback if correct or partially correct, plus explanation.
                    - Constructive feedback and correct answer if incorrect.

                8. **Language:** English only.

                ---------------------
                Query: {query}
                ---------------------
                Answer:
        """

        self.evaluation_prompt = """
        Evaluate the following user answer based on the question asked previously.

        Question:
        {question}

        User Answer:
        {user_answer}

        Your task:
            1. Assess the answer for relevance, accuracy, completeness, and clarity.
            2. Assign a **numerical grade from 0 to 100**. Round to the nearest whole number.
            3. Provide constructive feedback and explain **exactly why you assigned this grade**.
            4. Use the following strict output format:
                - Check if the answer is relevant and correct regarding the question.
                - Provide constructive feedback.
                - If correct or partially correct → positive feedback + short explanation.
                - If incorrect → constructive feedback + correct answer.
                - Answer in English.
        
        """

    def _setup_llm(self):
        """Initialize LLM based on provider (Ollama or Azure OpenAI)."""
        if self.llm_provider == 'azure':
            return AzureOpenAI(
                api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            )
        else:  # ollama
            base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            return Ollama(
                model=self.llm_name,
                base_url=base_url,
                request_timeout=120.0
            )

    def _add_to_history(self, role, content):
        """Add a message to conversation history and enforce size limit."""
        self.conversation_history.append({"role": role, "content": content})

        # Enforce maximum history size (keep most recent turns)
        max_messages = self.max_history_turns * 2  # Each turn = user + assistant
        if len(self.conversation_history) > max_messages:
            # Remove oldest messages to stay within limit
            self.conversation_history = self.conversation_history[-max_messages:]

    def _get_conversation_context(self):
        """Format conversation history based on provider."""
        if self.llm_provider == 'azure':
            # Azure OpenAI expects plain dicts
            return self.conversation_history.copy()
        else:  # ollama
            # Ollama expects ChatMessage objects
            return [ChatMessage(role=msg["role"], content=msg["content"]) for msg in self.conversation_history]

    def clear_history(self):
        """Manually clear conversation history (e.g., on difficulty change)."""
        self.conversation_history = []
        self.last_question = None

    def generate_context(self, query):
        result = self.retriever.search(query)
        context = [dict(data) for data in result]
        combined_prompt = []

        for entry in context:
            context_str = entry["payload"]["context"]
            combined_prompt.append(context_str)

        return "\n\n---\n\n".join(combined_prompt)

    def query(self, query):
        """
        Questo metodo gestisce:
        - Prima interazione: genera una domanda aperta (with conversation history context)
        - Seconda interazione: valuta la risposta rispetto all'ultima domanda (with conversation history context)
        """
        # Se abbiamo già una domanda precedente, allora la nuova query è una risposta
        if self.last_question:
            evaluation_prompt = self.evaluation_prompt.format(
                question=self.last_question,
                user_answer=query
            )

            # Build messages with conversation history
            if self.llm_provider == 'azure':
                messages = [{"role": "system", "content": "You are a helpful evaluator with access to conversation history."}]
                messages.extend(self._get_conversation_context())
                messages.append({"role": "user", "content": evaluation_prompt})

                response = self.llm.chat.completions.create(
                    model=self.llm_name,
                    messages=messages
                )
                assistant_reply = response.choices[0].message.content
            else:  # ollama
                messages = [ChatMessage(role="system", content="You are a helpful evaluator with access to conversation history.")]
                messages.extend(self._get_conversation_context())
                messages.append(ChatMessage(role="user", content=evaluation_prompt))

                response = self.llm.chat(messages=messages)
                assistant_reply = response.message.content

            # Add user answer and evaluation to history (persistent memory)
            self._add_to_history("user", query)
            self._add_to_history("assistant", assistant_reply)

            # Reset question tracker after evaluation
            self.last_question = None

            return assistant_reply

        # Altrimenti è la prima domanda → generiamo una open-ended question
        context = self.generate_context(query)
        difficulty_instructions = self.difficulty_instructions.get(self.difficulty, self.difficulty_instructions["Medium"])
        prompt = self.qa_prompt_tmpl_str.format(
            context=context,
            query=query,
            difficulty=self.difficulty,
            difficulty_instructions=difficulty_instructions
        )

        # Build messages with conversation history
        if self.llm_provider == 'azure':
            messages = [{"role": "system", "content": "You are a helpful assistant with access to conversation history."}]
            messages.extend(self._get_conversation_context())
            messages.append({"role": "user", "content": prompt})

            response = self.llm.chat.completions.create(
                model=self.llm_name,
                messages=messages
            )
            assistant_reply = response.choices[0].message.content
        else:  # ollama
            messages = [ChatMessage(role="system", content="You are a helpful assistant with access to conversation history.")]
            messages.extend(self._get_conversation_context())
            messages.append(ChatMessage(role="user", content=prompt))

            response = self.llm.chat(messages=messages)
            assistant_reply = response.message.content


        # Add to persistent history
        self._add_to_history("user", query)
        self._add_to_history("assistant", assistant_reply)

        # Salva l'ultima domanda dell'assistente per la valutazione futura
        self.last_question = assistant_reply

        return assistant_reply
    