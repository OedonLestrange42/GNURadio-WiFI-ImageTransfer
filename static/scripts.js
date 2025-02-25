document.addEventListener('DOMContentLoaded', (event) => {
    const socket = io();

    socket.on('image_update', function(data) {
        const imagePreview = document.getElementById('imagePreview');
        imagePreview.src = data.image;
    });
});