document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('textInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultSection = document.getElementById('resultSection');
    const originalText = document.getElementById('originalText');
    const processedText = document.getElementById('processedText');
    const sentimentDisplay = document.getElementById('sentimentDisplay');
    const exampleButtons = document.querySelectorAll('.example-btn');

    // Function to preprocess text (matching the Python preprocessing)
    function preprocessText(text) {
        // Convert to lowercase
        text = text.toLowerCase();
        
        // Remove URLs
        text = text.replace(/https?:\/\/\S+|www\.\S+/g, '');
        
        // Remove HTML tags
        text = text.replace(/<.*?>/g, '');
        
        // Remove user mentions
        text = text.replace(/@\w+/g, '');
        
        // Remove hashtags (keeping the text after #)
        text = text.replace(/#(\w+)/g, '$1');
        
        // Remove punctuation
        text = text.replace(/[^\w\s]/g, '');
        
        // Remove numbers
        text = text.replace(/\d+/g, '');
        
        // Remove extra whitespace
        text = text.replace(/\s+/g, ' ').trim();
        
        return text;
    }

    // Function to show error message
    function showError(message) {
        sentimentDisplay.className = 'sentiment-display sentiment-error';
        sentimentDisplay.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${message}`;
        resultSection.style.display = 'block';
    }

    // Function to analyze sentiment
    async function analyzeSentiment(text) {
        try {
            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Network response was not ok');
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error:', error);
            throw error;
        }
    }

    // Function to display results
    function displayResults(original, processed, sentiment) {
        originalText.textContent = original;
        processedText.textContent = processed;
        
        // Clear previous sentiment classes
        sentimentDisplay.className = 'sentiment-display';
        
        // Add appropriate sentiment class and content
        if (sentiment === 4) {
            sentimentDisplay.classList.add('sentiment-positive');
            sentimentDisplay.innerHTML = '<i class="fas fa-smile"></i> Positive Sentiment';
        } else if (sentiment === 0) {
            sentimentDisplay.classList.add('sentiment-negative');
            sentimentDisplay.innerHTML = '<i class="fas fa-frown"></i> Negative Sentiment';
        } else {
            sentimentDisplay.classList.add('sentiment-neutral');
            sentimentDisplay.innerHTML = '<i class="fas fa-meh"></i> Neutral Sentiment';
        }
        
        // Show result section with animation
        resultSection.style.display = 'block';
    }

    // Event listener for analyze button
    analyzeBtn.addEventListener('click', async () => {
        const text = textInput.value.trim();
        
        if (!text) {
            showError('Please enter some text to analyze.');
            return;
        }

        try {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
            
            const processed = preprocessText(text);
            const result = await analyzeSentiment(text);
            displayResults(text, processed, result.sentiment);
        } catch (error) {
            showError(error.message || 'An error occurred while analyzing the text. Please try again.');
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = '<i class="fas fa-magic"></i> Analyze Sentiment';
        }
    });

    // Event listeners for example buttons
    exampleButtons.forEach(button => {
        button.addEventListener('click', () => {
            const exampleText = button.dataset.text;
            textInput.value = exampleText;
            analyzeBtn.click();
        });
    });
}); 