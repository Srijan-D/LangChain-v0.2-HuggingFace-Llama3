document.getElementById('queryForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = document.getElementById('query').value;
    const responseElement = document.getElementById('response');
    responseElement.textContent = 'Loading...';

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.text();
        responseElement.textContent = result;
    } catch (error) {
        responseElement.textContent = 'Error: ' + error.message;
    }
});
