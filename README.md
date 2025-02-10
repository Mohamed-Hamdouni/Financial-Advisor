# AI Financial Advisor Pro

An advanced AI-powered financial analysis and investment recommendation system.

## Features

- **Real-time Market Data Analysis:** Access up-to-date market information for informed decision-making.
- **Document Processing (PDF Reports):** Automatically extract and analyze data from financial reports in PDF format.
- **Multi-Agent Investment Recommendations:** Leverage a sophisticated AI pipeline for personalized investment advice.
- **Interactive Web Interface:** A user-friendly platform for easy navigation and analysis.
- **Customizable Settings:** Tailor the AI's behavior with adjustable parameters like model choice and risk tolerance.
- **Historical Performance Analysis:** Visualize past performance with interactive charts and graphs.
- **Document Library:** Easily manage and access your financial documents.

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Mohamed-Hamdouni/Financial-Advisor.git
    cd financial-advisor
    ```

2.  **Create a Virtual Environment & Install Dependencies:**

    ```bash
    bash setup.sh
    ```

3.  **Set Up Environment Variables:**

    -   Create a `.env` file in the root directory.
    -   Add your API keys:

        ```properties
        OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
        NEWSAPI_KEY="YOUR_NEWSAPI_KEY"
        FINANCE_API_KEY="YOUR_FINANCE_API_KEY"
        ```

    -   Replace `"YOUR_API_KEY"` with your actual API keys.

## Running the Application

1.  **Start the Application:**

    ```bash
    bash start.sh
    ```

2.  **Access the Interface:**

    -   Open your web browser and go to the URL provided in the terminal (usually `http://localhost:8501`).

## Interface Overview

### Sidebar Navigation

-   **Dashboard:** View key metrics, historical performance charts, and recent market insights.
-   **Analysis:** Perform AI-driven investment analysis with customizable queries and stock symbols.
-   **Documents:** Manage and access your financial documents in PDF format.
-   **Settings:** Configure application settings such as model choice and risk tolerance.

### Dashboard

-   **Key Metrics:**
    -   Documents Analyzed: Number of PDF reports processed.
    -   Market Coverage: Percentage of the market analyzed.
    -   AI Confidence: Confidence level of the AI's recommendations.
-   **Historical Performance Analysis:**
    -   Quarterly Revenue Growth: Line chart visualizing revenue growth for major companies.
    -   ROI Comparison: Bar chart comparing return on investment for different companies and quarters.
-   **Recent Market Insights:**
    -   Expander sections displaying recent AI-driven market analyses.

### Analysis

-   **AI Investment Analysis:**
    -   Text area for entering custom investment queries.
    -   Dropdown menu for selecting stock symbols.
    -   Progress bar and status updates during analysis.
    -   Display of analysis results and previous analyses.

### Documents

-   **Document Library:**
    -   List of available PDF documents.
    -   Download links for each document.
    -   Categorization of documents into quarterly reports and other documents.

### Settings

-   **Model Configuration:**
    -   Language Model: Select between `gpt-3.5-turbo` and `gpt-4`.
    -   Creativity Level: Adjust the temperature to control the AI's creativity.
-   **Analysis Settings:**
    -   Analysis Depth: Choose between `Quick`, `Standard`, and `Deep` analysis.
    -   Risk Tolerance: Set your risk tolerance to `Conservative`, `Moderate`, or `Aggressive`.

## Troubleshooting

-   **Missing Dependencies:**
    -   If you encounter import errors, ensure all dependencies are installed correctly using `setup.sh`.
-   **API Key Issues:**
    -   Double-check that your API keys are correctly entered in the `.env` file.
-   **Database Initialization Errors:**
    -   Ensure the `data` and `vector_db` directories have the correct permissions (755).
    -   If the database fails to initialize, check the terminal logs for detailed error messages.
