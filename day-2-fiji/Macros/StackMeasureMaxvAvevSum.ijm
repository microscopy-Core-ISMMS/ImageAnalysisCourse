/*
 * lazy macro to show the difference between measuring Z-stack projections with
 * Maximum Projection
 * Average Projection
 * Sum Projection
 * Dale Moulding Jan 2021 
 */

// empty ROI manager & clear results:
run("Clear Results");
count=roiManager("count");	
		if (count != 0) {
			roiManager("Deselect");
			roiManager("delete");
		}

run("Organ of Corti (4D stack)");
run("Duplicate...", "title=Ch2 duplicate channels=2");
selectWindow("Ch2");
run("Z Project...", "projection=[Max Intensity]");
selectWindow("Ch2");
run("Z Project...", "projection=[Average Intensity]");
selectWindow("Ch2");
run("Z Project...", "projection=[Sum Slices]");
selectWindow("MAX_Ch2");
doWand(182, 104, 2000.0, "Legacy");
roiManager("Add");
doWand(200, 99, 2000.0, "Legacy");
roiManager("Add");
doWand(218, 94, 2000.0, "Legacy");
roiManager("Add");
doWand(236, 89, 2000.0, "Legacy");
roiManager("Add");
doWand(254, 86, 2000.0, "Legacy");
roiManager("Add");
run("Set Measurements...", "area mean display redirect=None decimal=3");
roiManager("Measure");
selectWindow("AVG_Ch2");
roiManager("Measure");
selectWindow("SUM_Ch2");
roiManager("Measure");
// make some Grpahs:
 Plot.create("MaxP", "{1,2,3,4,5}", "Y");
   yValues = newArray(getResult("Mean", 0), getResult("Mean", 1), getResult("Mean", 2), getResult("Mean", 3), getResult("Mean", 4));
   Plot.setFrameSize(400, 300);
   Plot.setLimits(-0.5, yValues.length-0.5, 0, 4000);
   Plot.setFontSize(18);
   Plot.setLineWidth(2);
   Plot.setColor("blue", "#bbbbff");
   Plot.add("bars",  yValues);
   if (getVersion()>="1.51v") { // draw yValues:
      code = "code: setFont('sanserif',14*s,'bold anti');drawString(''+d2s(yval,1),x-12*s,y-4*s);";
      Plot.add(code, yValues);
   Plot.show;

    Plot.create("AveP", "{1,2,3,4,5}", "Y");
   yValues = newArray(getResult("Mean", 5), getResult("Mean", 6), getResult("Mean", 7), getResult("Mean", 8), getResult("Mean", 9));
   Plot.setFrameSize(400, 300);
   Plot.setLimits(-0.5, yValues.length-0.5, 0, 4000);
   Plot.setFontSize(18);
   Plot.setLineWidth(2);
   Plot.setColor("blue", "#bbbbff");
   Plot.add("bars",  yValues);
   if (getVersion()>="1.51v") { // draw yValues:
      code = "code: setFont('sanserif',14*s,'bold anti');drawString(''+d2s(yval,1),x-12*s,y-4*s);";
      Plot.add(code, yValues);
   Plot.show;

 Plot.create("SumP", "{1,2,3,4,5}", "Y");
   yValues = newArray(getResult("Mean", 10), getResult("Mean", 11), getResult("Mean", 12), getResult("Mean", 13), getResult("Mean", 14));
   Plot.setFrameSize(400, 300);
   Plot.setLimits(-0.5, yValues.length-0.5, 0, 50000);
   Plot.setFontSize(18);
   Plot.setLineWidth(2);
   Plot.setColor("blue", "#bbbbff");
   Plot.add("bars",  yValues);
   if (getVersion()>="1.51v") { // draw yValues:
      code = "code: setFont('sanserif',14*s,'bold anti');drawString(''+d2s(yval,1),x-12*s,y-4*s);";
      Plot.add(code, yValues);
   Plot.show;