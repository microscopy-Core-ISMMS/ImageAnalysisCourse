/*  macro by Dale Moulding April 2019
 *  Helps in measuring western blots as per the image analysis course run at UCL GOS Institute of Child Health.
 *  Open your blot, run the macro, draw an ROI (rectangle, line or segmented line with a set line width)
 *  Follow prompts. The final image is a graph of the overlayed plots. 
 *  You may need to define the edges of each band with a drawn line.
 *  Use magic wand to make an ROI for each band and press "M" on keyboard to measure areas.
 */
run("Set Measurements...", "area redirect=None decimal=3");
waitForUser("Draw ROI around bands \n \n    ** Then press OK **");
run("Plot Profile");
rename("Bands");
lower = 0 // you can change this value to make the graph start at a value other than 0. Change if needed.
upper = 255 // defaults are 0 (lower) to 255 (upper).
Plot.setLimits(NaN,NaN,lower,upper); 
Plot.setFormatFlags("11001100001111");
waitForUser("Draw ROI around background, same width as bands ROI \n \n                            ** Then press OK **");
run("Plot Profile");
rename("Background");
Plot.setLimits(NaN,NaN,lower,upper);
Plot.setFormatFlags("11001100001111");
imageCalculator("Min create", "Bands","Background");
close("Bands");
close("Background");
selectWindow("Result of Bands");
rename("Click eack peak / trough with Wand to Measure Areas");
setTool("line");
run("Line Width...", "line=1");
setTool("wand");
waitForUser("*****    Separate lanes with a drawn line if needed   *****\n \nUse Wand to select each band and measure (press M) \n \n*****           Results will be in the results table          *****");