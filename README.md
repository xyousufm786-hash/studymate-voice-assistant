# StudyMate: a voice-enabled AI student assistant

A web app that helps university students plan, understand, practise and prepare for their careers. You can type or talk to it, and it can read its answers aloud. It runs locally with Ollama, so it's free and private.

## Modes
| Mode | What it does |
|---|---|
| Ask Anything | Academic and student-life questions |
| Study Planner | Builds a timetable from your subjects, exam dates and available hours |
| Explain a Topic | Definition, how it works, an example, a common mistake, and a check question |
| Quiz Me | 5 multiple-choice questions with an answer key, and marks your replies |
| Career & CV | CV, internship and interview help |

## Features
- 🎤 Voice input and 🔊 voice output through the browser's Web Speech API (English US/UK/India and Malay)
- Answers stream live, with chat memory for follow-up questions
- Input validation and clear error messages (Ollama offline, model missing, message too long)
- Light and dark mode, works on phones

## Run it
```bash
ollama pull llama3.2:1b
pip install -r requirements.txt
python app.py
```
Open **http://127.0.0.1:5000** in Chrome or Edge, since Firefox doesn't support voice input. Or run all cells in `StudyMate_Flask_Voice_Chatbot.ipynb`.

**Stack:** Flask · Ollama (Llama 3.2 1B) · HTML/CSS/JavaScript · Web Speech API
