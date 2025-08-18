import ollama

class Llama3Classifier:
    def __init__(self, logger, model: str = "llama-guard3:1b"):
        self.model = model
        self.logger = logger
        self.dict_hazard_categories = {
            "S1": "Violent Crimes",
            "S2": "Non-Violent Crimes",
            "S3": "Sex-Related Crimes",
            "S4": "Child Sexual Exploitation",
            "S5": "Defamation",
            "S6": "Specialized Advice",
            "S7": "Privacy",
            "S8": "Intellectual Property",
            "S9": "Indiscriminate Weapons",
            "S10": "Hate",
            "S11": "Suicide & Self-Harm",
            "S12": "Sexual Content",
            "S13": "Elections"
        }

    def interpret_response(self, response: str):
        lines = response.strip().splitlines()

        if len(lines) >= 2:
            safety_label = lines[0].strip()
            hazard_code = lines[1].strip()
            hazard_name = self.dict_hazard_categories.get(hazard_code, "Unknown Category")

            return f"{safety_label.capitalize()}, Reason Code: {hazard_code}, Description: {hazard_name}"
            print(f"{safety_label}, {hazard_code} : {hazard_name}")
        else:
            return response.capitalize()
            print("⚠️ Unexpected response format:", response)


    def send_prompt(self, prompt: str) -> str:
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response["message"]["content"]
        except Exception as e:
            self.logger.error(f"❌ Error sending prompt: {e}")
            return f"⚠️ Error: {str(e)}"


    def classify_prompt(self, prompt: str) -> str:
        self.logger.info(f"🔍 Classifying prompt with {self.model}: {prompt[:100]}")
        return self.interpret_response(self.send_prompt(prompt=prompt))

    def classify_response(self, response_text: str) -> str:
        self.logger.info(f"🔍 Classifying response with {self.model}: {response_text[:100]}")
        return self.interpret_response(self.send_prompt(prompt=response_text))