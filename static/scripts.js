$(document).ready(function(){
    $("#loading").hide();

    $('#send').click(function(evt){
        sendRequest(base_data);
    });

    $('#uload').click(function(evt) {
        $('#fileinput').focus().trigger('click');
    });

    $("#fileinput").change(function(){
        if (this.files && this.files[0]){
            var reader = new FileReader();
            reader.onload = function (e){
                var url = e.target.result;
                var img = new Image();
                img.crossOrigin = 'Anonymous';
                img.onload = function(){
                    var canvas = document.createElement('CANVAS');
                    var ctx = canvas.getContext('2d');
                    canvas.height = this.height;
                    canvas.width = this.width;
                    ctx.drawImage(this, 0, 0);
                    base_data = canvas.toDataURL('image/jpeg', 1.0).split(',')[1]; // Remove the data:image/jpeg;base64, part
                    canvas = null;
                };
                img.src = url;
                $('#photo').attr('src', url);
                $('#photo').show();
                $('#video').hide();
            }
            reader.readAsDataURL(this.files[0]);
        }
    });
});


console.log(typeof dataURLToBlob);
function sendRequest(base64Data){
    function dataURLToBlob(dataURL) {
        var binary = atob(dataURL.split(',')[1]);
        var array = [];
        for (var i = 0; i < binary.length; i++) {
            array.push(binary.charCodeAt(i));
        }
        return new Blob([new Uint8Array(array)], {type: 'image/jpeg'});
    }
    if(base64Data !== "" && base64Data !== null){
        var url = $("#url").val();
        $("#loading").show();
        
        var blob = dataURLToBlob('data:image/jpeg;base64,' + base64Data);
        var formData = new FormData();
        formData.append('file', blob, 'uploaded_image.jpg');
        $.ajax({
            url : url,
            type: "POST",
            cache: false,
            async: true,
            crossDomain: true,
            processData: false,
            contentType: false,
            data: formData,
            success: function(res) {
                $(".res-part").html("");
                $(".res-part2").html("");
                try {
                    var prediction = res.prediction;
                    if (prediction) {
                        $(".res-part").html("<pre>" + prediction + "</pre>");
                    } else {
                        $(".res-part").html("<pre>No prediction received.</pre>");
                    }
                } catch (e) {
                    $(".res-part").html("<pre>Error: " + e.toString() + "</pre>");
                }
                $("#loading").hide();
            },
            error: function(xhr, status, error) {
                $(".res-part").html("<pre>Error: " + error + "</pre>");
                $("#loading").hide();
            }
        });
    }
}

