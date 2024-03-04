function classifyText() {
    var inputText = document.getElementById("editor").innerText;
    fetch('/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({text: inputText}),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("output").innerText = JSON.stringify(data);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
