//selecting all required elements
const dropArea = document.querySelector(".drag-area"),
dragText = dropArea.querySelector("header"),
button = dropArea.querySelector("button"),
input = dropArea.querySelector("input");
let file; //this is a global variable and we'll use it inside multiple functions


button.onclick = ()=>{
  input.click(); //If user click on the button then the input also clicked
}

input.addEventListener("change", function(){
  //getting user select file ans [0] this means if user select multiple files then
  //we'll select only the first one
   file = this.file[0];
   showFile(); //calling function
   dropArea.classList.add("active");
});


//If user Drag File over DropArea
dropArea.addEventListener("dragover", (event)=>{
  event.preventDefault();//Preventing from default behavior
  dropArea.classList.add("active");
  dragText.textContent = "Release to Upload File";
});

//If user Drag File from DropArea
dropArea.addEventListener("dragleave", ()=>{
  dropArea.classList.remove("active");
  dragText.textContent = "Drag & Drop to Upload File";
});

//If user Drop File on DropArea
dropArea.addEventListener("drop", (event)=>{
  event.preventDefault();//Preventing from default behavior
  //getting user select file ans [0] this means if user select multiple files then
  //we'll select only the first one
  file = event.dataTransfer.files[0];
   showFile(); //calling function
});

function showFile(){
  let fileType = file.type;

  let validExtensions = ["audio/wav", "audio/wave", "audio/mpeg"]; //adding some valid image extensions in arrt
  if(validExtensions.includes(fileType)){ //If user selected Audion file
    let fileReader = new FileReader(); //creating ner FilwReader object
    fileReader.onload = ()=>{
      let fileURL = fileReader.result; //passing user file resource in file URL variable
      console.log(fileURL);
      let audioTag = '<audio src="${fileURL}" alt="">'; //creating an audio tag and passing user selected file source inside src attribute
      dropArea.innerHTML = audioTag; //adding that created audioTag inside dropArea container
    }
    fileReader.readAsDataURL(file);
  }else{
    alert("This is not an Audio File!");
    dropArea.classList.romove("active");
    dragText.textContent = "Drag & Drop to Upload File";
  }
}
