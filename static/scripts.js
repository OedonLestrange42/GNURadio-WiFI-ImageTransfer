document.addEventListener("DOMContentLoaded", function() {
    var socket = io.connect('http://' + document.domain + ':' + location.port);

    socket.on('update_image', function(data) {
        var img = document.getElementById('reconstructedImage');
        img.src = 'data:image/jpeg;base64,' + data.image;
    });
});