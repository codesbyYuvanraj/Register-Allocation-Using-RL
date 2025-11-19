# 🚀 Register Allocation Optimization with Deep Reinforcement Learning

## 📋 Overview

This project implements an advanced **Register Allocation System** using **Deep Reinforcement Learning** to optimize compiler register allocation. The system combines neural networks with traditional graph coloring algorithms to achieve superior performance over heuristic methods.

> **🎯 Goal**: Train an AI agent that learns optimal register allocation strategies through reinforcement learning

---
## The code files we have submitted:
- `Batch 5 TOACD Project.ipynb`
- `Batch 5 TOACD Project.py`

**Kindly use `Batch 5 TOACD Project.ipynb` file for evalulating purpose in **Google Colab** for smooth Experience**

---
## 🏗️ Project Structure

This Jupyter notebook contains **4 main cells** that should be executed in sequence:

### 🧠 Cell 1: Core RL Model Implementation
**File**: `Register_Allocation_RL_Model.ipynb` (First Main Cell)

**What it does**:
- Defines the `RegAllocRL` class with Deep Q-Network architecture
- Implements reinforcement learning for register allocation decisions
- Includes training routines with curriculum learning
- Provides evaluation methods for model performance

**Expected Output**:
```text
Enhanced RL Allocator on cuda: 5 hidden layers, 128 nodes
Network architecture: 16→128 → 128→128 → 128→128 → 128→64 → 64→300
Training for 5900 episodes | graph_range=(30, 60) | LR=0.0005
...
Training completed.
```

### 🔬 Cell 2: Model Evaluation & Performance Analysis
**File**: `Register_Allocation_RL_Model.ipynb` (First Mini Cell)

**What it does**:
- Tests the trained RL agent on sample problems
- Compares performance against traditional heuristic methods
- Measures strategic accuracy and computational efficiency

**Code to run**:
```python
agent.model_evaluation(num_tests=10, test_cases=[(50, 4, 0.3)])
```

**Expected Output**:
```text
Evaluating on 10 problems (size=50, K=4, p=0.3)
   RL Performance:
   Strategic Accuracy: 0.852 | Traditional Success: 0.734 | Time: 0.0456s
   Heuristic Baseline:
   Strategic Accuracy: 0.812 | Traditional Success: 0.698 | Time: 0.0123s
   Improvement: +0.040 Strategic Accuracy, +0.036 Success Rate
```

### 🎨 Cell 3: Interactive GUI Implementation
**File**: `Register_Allocation_RL_Model.ipynb` (Second Main Cell)

**What it does**:
- Creates comprehensive visual interface for register allocation
- Implements real-time graph visualization and animation
- Provides performance analytics and comparison tools
- Supports both automatic and manual graph input

### 🖥️ Cell 4: GUI Launch & Interaction
**File**: `Register_Allocation_RL_Model.ipynb` (Second Mini Cell)

**What it does**:
- Initializes and displays the interactive GUI
- Connects the trained RL agent to the visualization interface
- Enables user interaction with all GUI features

**Code to run**:
```python
gui = RegAllocGUI(agent=agent)
```

---

## 🚀 Quick Start Guide

### Step 1: Run Core RL Model Cell
Execute the first main cell to initialize and train the reinforcement learning agent. This may take several minutes depending on your hardware.

### Step 2: Evaluate Model Performance
Run the first mini cell to test the trained agent and see performance metrics compared to traditional methods.

### Step 3: Launch GUI Interface
Execute the second main cell to load the graphical user interface components.

### Step 4: Start Interactive Session
Run the second mini cell to launch the GUI and begin interactive exploration.

---

## 🎯 What Each Component Does

### 🤖 Core RL Model
The **brain** of the system - a Deep Q-Network that learns to make optimal register allocation decisions by:

- Converting interference graphs into numerical features
- Learning which nodes to spill through trial and error
- Improving decisions based on strategic rewards
- Adapting to different graph structures and sizes

### 🎨 GUI Interface
The **visual control center** that lets you:

- Generate random interference graphs with custom parameters
- Evaluate the RL model's allocation decisions in real-time
- Compare RL performance against traditional heuristic methods
- Visualize the allocation process with animated step-by-step explanations
- Analyze performance trends and computational efficiency

---
## 🖥️ GUI Navigation & Interface Guide

### Overview of Interactive Interface
The Register Allocation GUI provides a comprehensive visualization and analysis platform with four specialized tabs, each designed for specific analytical workflows. The interface features real-time performance tracking, advanced visualization, and detailed analytics.

### 📊 Tab 1: Interactive Evaluation (🧪 EVAL)

#### Graph Configuration Panel
- **Node Count Slider**: Adjust graph complexity (10-500 nodes)
- **Register Count (K)**: Set available registers (2-20 registers) 
- **Edge Probability**: Control graph connectivity (0.05-1.0)
- **Complexity Indicator**: Real-time complexity assessment

#### Execution Workflow
```python
# Primary Analysis Sequence
1. SET_PARAMETERS() → 2. GENERATE_GRAPH() → 3. EVALUATE_MODEL() → 4. COMPARE_METHODS()
```

**Core Functions**:
- **🎲 GENERATE GRAPH**: Creates random interference graphs with specified parameters
- **🚀 EVALUATE MODEL**: Executes RL agent with real-time visualization and performance metrics
- **⚖️ COMPARE METHODS**: Runs comparative analysis against traditional heuristic algorithms
- **🎬 RENDER ALLOCATION VIDEO**: Generates MP4 video of complete allocation process

#### Real-time Visualization Features
- **Dynamic Node Coloring**: 
  - 🟢 **Allocated Nodes**: Successfully assigned to registers
  - 🔴 **Spilled Nodes**: Moved to memory due to register pressure
  - 🟡 **Current Node**: Actively being processed
  - 🔵 **Simplified Nodes**: Removed during graph simplification
  - 🟣 **Interfering Nodes**: Neighbors causing register conflicts

- **Phase-based Visualization**:
  - **Simplification Phase**: Low-degree node removal
  - **Spilling Phase**: Strategic spill decisions
  - **Allocation Phase**: Register assignment process
  - **Final Results**: Complete allocation mapping

### 📝 Tab 2: Manual Graph Input (✏️ MANUAL)

#### Custom Graph Interface
- **Text Input Area**: Direct adjacency list specification
- **Example Templates**: Pre-configured graph structures
- **Syntax Validation**: Automatic format checking

**Supported Format**:
```
node_id: neighbor1, neighbor2, neighbor3
0: 1, 2, 3
1: 0, 2, 4
2: 0, 1, 3
```

**Workflow Controls**:
- **📋 LOAD EXAMPLE**: Preloaded complex graph examples
- **🔍 EVALUATE CUSTOM**: Execute RL model on custom graphs
- **📈 ADVANCED ANALYSIS**: Detailed structural analysis

### 📋 Tab 3: Results Dashboard (📊 RESULTS)

#### Comprehensive Output Display
- **Performance Metrics**: Success rates and strategic accuracy
- **Allocation Maps**: Detailed register assignment tables
- **Spill Analysis**: Spilled nodes with cost considerations
- **Video Export**: Downloadable MP4 allocation animations

#### Output Structure
```python
{
    "strategic_accuracy": 0.852,
    "traditional_success": 0.734, 
    "computation_time": 0.0456,
    "allocation_map": {"node_1": "R0", "node_2": "SPILL", ...},
    "spill_decisions": 12,
    "conflicts_resolved": 45
}
```

### 📈 Tab 4: Performance Analytics (📈 ANALYTICS)

#### Advanced Metrics Dashboard
- **Time Range Selection**: Last 10/25/50/All evaluations
- **Metric Focus**: Success Rate, Strategic Accuracy, Improvement vs Heuristic, **Time Performance**
- **Historical Trends**: Performance evolution over time

#### Real-time Analytics Features
- **Performance Overview**:
  - Average success rates and accuracy
  - Improvement trends vs heuristic methods
  - Graph complexity analysis

- **Time Performance Analysis** (NEW):
  - RL vs Heuristic execution time comparison
  - Time efficiency ratios and trends
  - Computational overhead assessment

- **Comparative Analytics**:
  - Win rate statistics
  - Best improvement tracking
---

## 📊 Understanding the Output

### Core RL Model Generates:
- **Training Progress**: Learning curves and performance metrics
- **Model Checkpoints**: Saved neural networks (`*.pth` files)
- **Performance Tables**: Detailed comparison data
- **Evaluation Results**: Strategic accuracy and success rates

### GUI Interface Provides:
- **Real-time Visualization**: Interactive graph displays
- **Performance Dashboards**: Color-coded success metrics
- **Allocation Maps**: Detailed register assignment lists
- **Analytics History**: Trend analysis and improvement tracking
- **Video Exports**: Animated allocation process recordings

---

## 🔧 Technical Requirements

### Required Python Packages:
```python
torch >= 1.9.0          # Deep learning framework
networkx >= 2.6.3       # Graph operations and visualization
matplotlib >= 3.3.4     # Plotting and animation
numpy >= 1.21.0         # Numerical computations
ipywidgets >= 7.6.5     # Interactive GUI components
```

### Recommended Hardware:
- **GPU**: CUDA-capable GPU for faster training (optional but recommended)
- **RAM**: Minimum 4GB, 8GB+ for larger graphs
- **Storage**: 500MB free space for models and outputs
- **Browser**: Modern web browser for GUI display

---

## 📈 Expected Performance

### 🎯 What Makes This Special:
- **Learning-Based**: Gets smarter with more experience
- **Strategic Thinking**: Focuses on avoiding unnecessary spills
- **Real-Time Feedback**: Instant analysis of allocation decisions
- **Visual Learning**: See the AI's thought process step by step

### ⚡ Typical Improvements:
- **10-20% Better Decisions**: Fewer avoidable spills than traditional methods
- **5-15% More Success**: More nodes successfully allocated to registers
- **Adaptive Intelligence**: Works well on different types of graphs
- **Time Efficient**: Reasonable computation time for the quality gained

---

## 🔗 Repository Information

### 📁 GitHub Repository:

> *https://github.com/codesbyYuvanraj/Register-Allocation-Using-RL*

### 📂 Project Structure:
```text
register-allocation-using-rl/
└── Batch 5 TOACD Project.ipynb  # Complete project notebook
    ├── Core RL Model Implementation
    ├── Model Evaluation & Analysis
    ├── Interactive GUI Implementation
    └── GUI Launch & Interaction
```

---

## 🎉 Conclusion & Next Steps

### 🌟 What You've Accomplished:
By running this notebook, you've:
- Trained a smart AI that learns register allocation
- Compared its performance against traditional methods
- Visualized the process through an interactive interface
- Analyzed the results with comprehensive analytics

### 🚀 Where to Go From Here:
- Experiment with different graph sizes and complexities
- Test the model on your own custom interference graphs
- Extend the RL model with different architectures
- Integrate with real compiler frameworks
- Explore other compiler optimization problems

### 💡 Learning Outcomes:
- Understand how reinforcement learning applies to compiler optimization
- See the trade-offs between different register allocation strategies
- Learn how to visualize and analyze complex optimization processes
- Experience interactive machine learning model evaluation

---

<div align="center">

**Built with ❤️ using PyTorch, NetworkX, and Interactive Widgets**  


</div>

---
