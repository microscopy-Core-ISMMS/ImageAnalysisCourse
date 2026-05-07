/*
 * Lazy macro to show different filters...
 * Dale Moulding Jan 2021
 */
Dialog.create("Set filter size (not for Anistropic diffusion)");
Dialog.addNumber("Filter size...(except Anisotropic diffusion)", 2);
Dialog.show();
filtersize = Dialog.getNumber();

run("Organ of Corti (4D stack)");
run("Duplicate...", "title=Median duplicate");
run("Duplicate...", "title=Gaussian duplicate");
run("Duplicate...", "title=Mean duplicate");
run("Duplicate...", "title=Anisotropic duplicate");
run("Anisotropic Diffusion 2D", "number=20 smoothings=1 keep=20 a1=0.50 a2=0.90 dt=20 edge=5");
run("Stack to Hyperstack...", "order=xyczt(default) channels=4 slices=15 frames=1 display=Composite");
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


