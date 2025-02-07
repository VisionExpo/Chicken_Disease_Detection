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
                    base_data = canvas.toDataURL('image/jpeg', 1.0).replace(/^data:image.+;base64,/, '');
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

function sendRequest(base64Data){
    var type = "json";
    if(base64Data != "" || base64Data != null){
        if(type == "imgtobase"){
            $(".res-part").html("");
            $(".res-part").html(base64Data);
        }
        else if(type == "basetoimg"){
            var imageData = $("#imgstring").val();
            $(".res-part").html("");
            $(".res-part").append("<img src='data:image/jpeg;base64," + imageData + "' alt='' />");
        }
        else{
            var url = $("#url").val();
            $("#loading").show();
            $.ajax({
                url : url,
                type: "post",
                cache: false,
                async: true,
                crossDomain: true,
                headers: {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin':'*'
                },
                data:JSON.stringify({image:base64Data}),
                success: function(res){
                    $(".res-part").html("");
                    $(".res-part2").html("");
                    try{
                        var imageData = res[1].image;
                        if(imageData.length > 100){
                            if(imageData.length > 10){$(".res-part2").append("<img class='resp-img' src='data:image/jpeg;base64," + imageData + "' alt='' />");}
                        }
                    }catch(e){}
                    $(".res-part").html("<pre>" + JSON.stringify(res[0], undefined, 2) + "</pre>");
                    plotAccuracy(res[0].accuracy, res[0].validation_accuracy);
                    $("#loading").hide();
                }
            });
        }
    }
}
