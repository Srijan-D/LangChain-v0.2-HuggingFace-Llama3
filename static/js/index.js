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
        const result = await response.text(); //data is being passed as plain text / string from flask backend

        // Clear previous response
        responseElement.innerHTML = '';

        // Create and append the <h2> element
        const responseTitle = document.createElement('h2');
        responseTitle.textContent = 'Response:';

        responseElement.innerHTML = ''; // Clear previous response
        responseElement.appendChild(responseTitle); // Add the <h2> inside <pre>
        responseElement.appendChild(document.createTextNode(result));

    } catch (error) {
        responseElement.textContent = 'Error: ' + error.message;
    }
});
