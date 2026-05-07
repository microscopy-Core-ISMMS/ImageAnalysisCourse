//empty macro to deal with files in a folder

macro "Put a name here"{ // change the name to anything you like

// The next 3 lines asks for the folder of files, where ot save the results, then makes a list of the files.

dir1 = getDirectory("Input folder"); //select an input folder
dir2 = getDirectory("Choose a folder to save to"); //select an output folder.
list = getFileList(dir1); //make a list of the filenames

setBatchMode(true); //turn on batch mode so it runs in the background

// repeats the macro for every file in the folder using a for loop. 
	for (i=0; i<list.length; i++) {
	showProgress(i+1, list.length);
	filename = dir1 + list[i];
	
			if (endsWith(filename, "something")) { // change 'something' to the end of the filenames you want to open. In this case "Dapi.TIF"
			open(filename);
			Imagename = File.nameWithoutExtension;	// this lets you use the string "Imagename" in the macro to save files with the name of the input file
			// your code goes here

			// add commands from the macro recorder. 
			// do just the counts first.

			//then try and make an image of each counted file, and save it.
			//This can be done with merge channels, but will need the string Imagename written +Imagename+
			
			
			} // end of the processing of each file

	}// end of the for loop for each file
// add a bit here to save the summary window (see 2Dcilia macro)
exit("Macro done in "+i+" images"); // close the macro and display a window with number of images processed.
}