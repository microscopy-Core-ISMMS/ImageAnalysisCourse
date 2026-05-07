/*
 * Lazy macro to show different filters...
 * Dale Moulding Jan 2021
 */
Dialog.create("Set filter size (not for Anistropic diffusion)");
Dialog.addNumber("Filter size...(except Anisotropic diffusion)", 5);
Dialog.show();
filtersize = Dialog.getNumber();

run("Blobs (25K)");
run("Duplicate...", "title=Blobs&Noise");
//run("Salt and Pepper");
run("Add Specified Noise...", "standard=16");
run("Duplicate...", "title=Median");
run("Duplicate...", "title=Gaussian");
run("Duplicate...", "title=Mean");
run("Duplicate...", "title=Anisotropic");
run("Anisotropic Diffusion 2D", "number=20 smoothings=1 keep=20 a1=0.50 a2=0.90 dt=20 edge=5");
selectWindow("Gaussian");
run("Gaussian Blur...", "sigma="+filtersize);
rename("Gaussian"+filtersize);
selectWindow("Median");
run("Median...", "radius="+filtersize);
rename("Median"+filtersize);
selectWindow("Mean");
run("Mean...", "radius="+filtersize);
rename("Mean"+filtersize);
selectWindow("Anisotropic");
close();
run("Tile");


