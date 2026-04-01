# Smart Irrigation AI

### AI-powered irrigation that saves water, protects crops, and adapts to uncertain weather.

## 🌍 Problem Statement

Agriculture depends heavily on water, but traditional irrigation often wastes it.

Farmers face difficult questions every day:
- How much water should be used?
- Which field zone needs irrigation most?
- What if rain is coming soon?
- How can crop health be protected without overwatering?

In drought-prone and climate-uncertain regions, poor irrigation decisions can lead to:
- water waste
- reduced crop yield
- soil imbalance
- higher farming costs

## 💡 Solution

Smart Irrigation AI is an intelligent irrigation simulation system that helps optimize water usage based on environmental conditions.

It uses:
- temperature
- soil moisture
- rain forecast
- zone-based decision logic

The system simulates how an AI controller would make irrigation decisions while balancing:
- crop health
- water efficiency
- weather awareness

## ✨ Features

- AI-based irrigation decision simulation
- Zone-wise irrigation control
- Inputs for temperature, soil moisture, and rain forecast
- Reward-based performance evaluation
- Live Gradio dashboard
- Soil moisture trend graphs
- Water usage trend graphs
- Reward trend visualization
- Multi-scenario benchmarking
- Hugging Face Spaces deployment support

## ⚙️ How It Works

1. The user sets environmental conditions such as temperature, rainfall forecast, and soil moisture.
2. The simulator evaluates the current irrigation state.
3. The AI determines how much water should be applied to the selected zone.
4. The environment updates soil moisture and water usage.
5. Rewards are calculated based on efficiency and crop health.
6. Graphs display performance over time.
7. The system repeats this process across multiple task difficulties.

## 🧠 Tech Stack

- Python
- Gradio
- NumPy
- Matplotlib
- Hugging Face Spaces
- Docker

## 📊 Demo / Output

The system provides a visual simulation dashboard with:
- reward trend graphs
- soil moisture graphs
- water usage graphs
- final performance scores
- task-level benchmarking

This makes it easy for judges and users to understand:
- how the AI is making decisions
- how efficiently water is being used
- how stable the irrigation strategy is

## 🚀 Installation

Clone the repository:

```bash
git clone <your-repo-url>
cd <your-project-folder>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

## ▶️ Usage

1. Launch the Gradio app.
2. Adjust:
   - temperature
   - rain forecast
   - soil moisture
   - zone selection
3. Start the simulation.
4. Observe:
   - live outputs
   - performance metrics
   - irrigation graphs
5. Compare behavior across different task difficulties.

## 🌱 Real-World Impact

Smart Irrigation AI can help demonstrate how AI can support sustainable agriculture.

Potential impact:
- saves water in drought-prone regions
- reduces overwatering and underwatering risk
- improves crop health
- supports data-driven farming decisions
- scales toward IoT-enabled smart farming systems

## 🔮 Future Improvements

- Real-time IoT sensor integration
- Mobile dashboard for farmers
- Weather API integration
- Reinforcement learning-based adaptive policies
- Farm-specific crop models
- Multi-zone optimization at larger scale

## 👤 Author

Built as a hackathon project focused on AI for sustainable agriculture.

## 👨‍💻 Team

- **Deepanshu Sirohi**
 

- **Sahas Rastohi**
  

- **Yashraj Gulyani**
  
