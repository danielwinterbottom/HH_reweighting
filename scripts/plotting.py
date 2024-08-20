import ROOT as R
import math

def GetRangeAroundMax(hist,frac=0.6):
  hist_scaled = hist.Clone()
  hist_scaled.Scale(1./hist.Integral(-1,-1))
  max_i=hist_scaled.GetMaximumBin()

  # get upper value but finding bin number for bin that changes ...
  tot_upper=0.
  for hi_i in range(max_i,hist.GetNbinsX()+1):
      tot_upper+=hist_scaled.GetBinContent(hi_i)
      if tot_upper>frac/2: break  

  
  tot_lower=0.
  for lo_i in range(max_i-1,0,-1):
      tot_lower+=hist_scaled.GetBinContent(lo_i)
      if tot_lower>frac/2: break

  lo_i+=1
  #print max_i, hi_i, lo_i
  #print hist.GetBinLowEdge(max_i), hist.GetBinLowEdge(hi_i), hist.GetBinLowEdge(lo_i)
  #print hist_scaled.Integral(lo_i,hi_i)
  return (lo_i,hi_i)

def SetTDRStyle():
    """Sets the PubComm recommended style

    Just a copy of <http://ghm.web.cern.ch/ghm/plots/MacroExample/tdrstyle.C>
    @sa ModTDRStyle() to use this style with some additional customisation.
    """
    # For the canvas:
    R.gStyle.SetCanvasBorderMode(0)
    R.gStyle.SetCanvasColor(R.kWhite)
    R.gStyle.SetCanvasDefH(600)  # Height of canvas
    R.gStyle.SetCanvasDefW(600)  # Width of canvas
    R.gStyle.SetCanvasDefX(0)    # POsition on screen
    R.gStyle.SetCanvasDefY(0)

    # For the Pad:
    R.gStyle.SetPadBorderMode(0)
    # R.gStyle.SetPadBorderSize(Width_t size = 1)
    R.gStyle.SetPadColor(R.kWhite)
    R.gStyle.SetPadGridX(False)
    R.gStyle.SetPadGridY(False)
    R.gStyle.SetGridColor(0)
    R.gStyle.SetGridStyle(3)
    R.gStyle.SetGridWidth(1)

    # For the frame:
    R.gStyle.SetFrameBorderMode(0)
    R.gStyle.SetFrameBorderSize(1)
    R.gStyle.SetFrameFillColor(0)
    R.gStyle.SetFrameFillStyle(0)
    R.gStyle.SetFrameLineColor(1)
    R.gStyle.SetFrameLineStyle(1)
    R.gStyle.SetFrameLineWidth(1)

    # For the histo:
    # R.gStyle.SetHistFillColor(1)
    # R.gStyle.SetHistFillStyle(0)
    R.gStyle.SetHistLineColor(1)
    R.gStyle.SetHistLineStyle(0)
    R.gStyle.SetHistLineWidth(1)

    R.gStyle.SetEndErrorSize(2)

    R.gStyle.SetMarkerStyle(20)

    # For the fit/function:
    R.gStyle.SetOptFit(1)
    R.gStyle.SetFitFormat('5.4g')
    R.gStyle.SetFuncColor(2)
    R.gStyle.SetFuncStyle(1)
    R.gStyle.SetFuncWidth(1)

    # For the date:
    R.gStyle.SetOptDate(0)

    # For the statistics box:
    R.gStyle.SetOptFile(0)
    R.gStyle.SetOptStat(0)
    # To display the mean and RMS:   SetOptStat('mr')
    R.gStyle.SetStatColor(R.kWhite)
    R.gStyle.SetStatFont(42)
    R.gStyle.SetStatFontSize(0.025)
    R.gStyle.SetStatTextColor(1)
    R.gStyle.SetStatFormat('6.4g')
    R.gStyle.SetStatBorderSize(1)
    R.gStyle.SetStatH(0.1)
    R.gStyle.SetStatW(0.15)

    # Margins:
    R.gStyle.SetPadTopMargin(0.05)
    R.gStyle.SetPadBottomMargin(0.13)
    R.gStyle.SetPadLeftMargin(0.16)
    R.gStyle.SetPadRightMargin(0.02)

    # For the Global title:
    R.gStyle.SetOptTitle(0)
    R.gStyle.SetTitleFont(42)
    R.gStyle.SetTitleColor(1)
    R.gStyle.SetTitleTextColor(1)
    R.gStyle.SetTitleFillColor(10)
    R.gStyle.SetTitleFontSize(0.05)

    # For the axis titles:
    R.gStyle.SetTitleColor(1, 'XYZ')
    R.gStyle.SetTitleFont(42, 'XYZ')
    R.gStyle.SetTitleSize(0.06, 'XYZ')
    # Another way to set the size?
    R.gStyle.SetTitleXOffset(0.9)
    R.gStyle.SetTitleYOffset(1.25)

    # For the axis labels:

    R.gStyle.SetLabelColor(1, 'XYZ')
    R.gStyle.SetLabelFont(42, 'XYZ')
    R.gStyle.SetLabelOffset(0.007, 'XYZ')
    R.gStyle.SetLabelSize(0.05, 'XYZ')

    # For the axis:

    R.gStyle.SetAxisColor(1, 'XYZ')
    R.gStyle.SetStripDecimals(True)
    R.gStyle.SetTickLength(0.03, 'XYZ')
    R.gStyle.SetNdivisions(510, 'XYZ')
    R.gStyle.SetPadTickX(1)
    R.gStyle.SetPadTickY(1)

    # Change for log plots:
    R.gStyle.SetOptLogx(0)
    R.gStyle.SetOptLogy(0)
    R.gStyle.SetOptLogz(0)

    # Postscript options:
    R.gStyle.SetPaperSize(20., 20.)

    R.gStyle.SetHatchesLineWidth(5)
    R.gStyle.SetHatchesSpacing(0.05)


def ModTDRStyle(width=600, height=600, t=0.06, b=0.12, l=0.16, r=0.04):
    """Modified version of the tdrStyle

    Args:
        width (int): Canvas width in pixels
        height (int): Canvas height in pixels
        t (float): Pad top margin [0-1]
        b (float): Pad bottom margin [0-1]
        l (float): Pad left margin [0-1]
        r (float): Pad right margin [0-1]
    """
    SetTDRStyle()

    # Set the default canvas width and height in pixels
    R.gStyle.SetCanvasDefW(width)
    R.gStyle.SetCanvasDefH(height)

    # Set the default margins. These are given as fractions of the pad height
    # for `Top` and `Bottom` and the pad width for `Left` and `Right`. But we
    # want to specify all of these as fractions of the shortest length.
    def_w = float(R.gStyle.GetCanvasDefW())
    def_h = float(R.gStyle.GetCanvasDefH())

    scale_h = (def_w / def_h) if (def_h > def_w) else 1.
    scale_w = (def_h / def_w) if (def_w > def_h) else 1.

    def_min = def_h if (def_h < def_w) else def_w

    R.gStyle.SetPadTopMargin(t * scale_h)
    # default 0.05
    R.gStyle.SetPadBottomMargin(b * scale_h)
    # default 0.13
    R.gStyle.SetPadLeftMargin(l * scale_w)
    # default 0.16
    R.gStyle.SetPadRightMargin(r * scale_w)
    # default 0.02
    # But note the new CMS style sets these:
    # 0.08, 0.12, 0.12, 0.04

    # Set number of axis tick divisions
    R.gStyle.SetNdivisions(510, 'XYZ')  # default 510

    # Some marker properties not set in the default tdr style
    R.gStyle.SetMarkerColor(R.kBlack)
    R.gStyle.SetMarkerSize(1.0)

    R.gStyle.SetLabelOffset(0.007, 'YZ')
    # This is an adhoc adjustment to scale the x-axis label
    # offset when we stretch plot vertically
    # Will also need to increase if first x-axis label has more than one digit
    R.gStyle.SetLabelOffset(0.005 * (3. - 2. / scale_h), 'X')

    # In this next part we do a slightly involved calculation to set the axis
    # title offsets, depending on the values of the TPad dimensions and
    # margins. This is to try and ensure that regardless of how these pad
    # values are set, the axis titles will be located towards the edges of the
    # canvas and not get pushed off the edge - which can often happen if a
    # fixed value is used.
    title_size = 0.05
    title_px = title_size * def_min
    label_size = 0.04
    R.gStyle.SetTitleSize(title_size, 'XYZ')
    R.gStyle.SetLabelSize(label_size, 'XYZ')

    R.gStyle.SetTitleXOffset(0.5 * scale_h *
                             (1.2 * (def_h * b * scale_h - 0.6 * title_px)) /
                             title_px)
    R.gStyle.SetTitleYOffset(0.5 * scale_w *
                             (1.2 * (def_w * l * scale_w - 0.6 * title_px)) /
                             title_px)

    # Only draw ticks where we have an axis
    R.gStyle.SetPadTickX(0)
    R.gStyle.SetPadTickY(0)
    R.gStyle.SetTickLength(0.02, 'XYZ')

    R.gStyle.SetLegendBorderSize(0)
    R.gStyle.SetLegendFont(42)
    R.gStyle.SetLegendFillColor(0)
    R.gStyle.SetFillColor(0)

    R.gROOT.ForceStyle()

def OnePad():
    pad = R.TPad('pad', 'pad', 0., 0., 1., 1.)
    pad.SetTicks(1)
    pad.Draw()
    pad.cd()
    result = [pad]
    return result

def TwoPadSplit(split_point, gap_low, gap_high):
    upper = R.TPad('upper', 'upper', 0., 0., 1., 1.)
    upper.SetBottomMargin(split_point + gap_high)
    upper.SetFillStyle(4000)
    upper.SetTicks(1)
    upper.Draw()
    lower = R.TPad('lower', 'lower', 0., 0., 1., 1.)
    lower.SetTopMargin(1 - split_point + gap_low)
    lower.SetFillStyle(4000)
    lower.Draw()
    upper.cd()
    result = [upper, lower]
    return result

def createAxisHists(n,src,xmin=0,xmax=499):
  result = []
  for i in range(0,n):
    res = src.Clone()
    res.Reset()
    res.SetTitle("")
    res.SetName("axis%(i)d"%vars())
    res.SetAxisRange(xmin,xmax)
    res.SetStats(0)
    result.append(res)
  return result

def PositionedLegend(width, height, pos, offset):
    o = offset
    w = width
    h = height
    l = R.gPad.GetLeftMargin()
    t = R.gPad.GetTopMargin()
    b = R.gPad.GetBottomMargin()
    r = R.gPad.GetRightMargin()
    if pos == 1:
        return R.TLegend(l + o, 1 - t - o - h, l + o + w, 1 - t - o, '', 'NBNDC')
    if pos == 2:
        c = l + 0.5 * (1 - l - r)
        return R.TLegend(c - 0.5 * w, 1 - t - o - h, c + 0.5 * w, 1 - t - o, '', 'NBNDC')
    if pos == 3:
        return R.TLegend(1 - r - o - w, 1 - t - o - h, 1 - r - o, 1 - t - o, '', 'NBNDC')
    if pos == 4:
        return R.TLegend(l + o, b + o, l + o + w, b + o + h, '', 'NBNDC')
    if pos == 5:
        c = l + 0.5 * (1 - l - r)
        return R.TLegend(c - 0.5 * w, b + o, c + 0.5 * w, b + o + h, '', 'NBNDC')
    if pos == 6:
        return R.TLegend(1 - r - o - w, b + o, 1 - r - o, b + o + h, '', 'NBNDC')
    if pos == 7:
        return R.TLegend(1 - o - w, 1 - t - o - h, 1 - o, 1 - t - o, '', 'NBNDC')

def DrawTitle(pad, text, align, scale=1):
    pad_backup = R.gPad
    pad.cd()
    t = pad.GetTopMargin()
    l = pad.GetLeftMargin()
    r = pad.GetRightMargin()

    pad_ratio = (float(pad.GetWh()) * pad.GetAbsHNDC()) / \
        (float(pad.GetWw()) * pad.GetAbsWNDC())
    if pad_ratio < 1.:
        pad_ratio = 1.

    textSize = 0.6
    textOffset = 0.4

    latex = R.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(R.kBlack)
    latex.SetTextFont(42)
    latex.SetTextSize(textSize * t * pad_ratio * scale)

    y_off = 1 - t + textOffset * t
    if align == 1:
        latex.SetTextAlign(11)
    if align == 1:
        latex.DrawLatex(l, y_off, text)
    if align == 2:
        latex.SetTextAlign(21)
    if align == 2:
        latex.DrawLatex(l + (1 - l - r) * 0.5, y_off, text)
    if align == 3:
        latex.SetTextAlign(31)
    if align == 3:
        latex.DrawLatex(1 - r, y_off, text)
    pad_backup.cd()

def CompareHists(hists=[],
             legend_titles=[],
             scale_factors=None,
             title="",
             ratio=True,
             log_y=False,
             log_x=False,
             ratio_range="0.7,1.3",
             custom_x_range=False,
             x_axis_max=4000,
             x_axis_min=0,
             custom_y_range=False,
             y_axis_max=4000,
             y_axis_min=0,
             x_title="",
             y_title="",
             extra_pad=0,
             norm_hists=False,
             plot_name="plot",
             label="",
             norm_bins=False,
             IncErrors=False,
             skipCols=0,
             lowerLeg=False,
             wideLeg=False):

    objects=[]
    R.gROOT.SetBatch(R.kTRUE)
    R.TH1.AddDirectory(False)
    widePlot=False
    if widePlot: ModTDRStyle(width=900,r=0.04, l=0.14)
    else:ModTDRStyle(r=0.04, l=0.14)

    colourlist=[R.kBlack,R.kBlue,R.kRed,R.kGreen+2,R.kMagenta,R.kCyan+1,R.kYellow+2,R.kViolet-5,R.kOrange,R.kCyan+3,R.kGray]

    colourlist = colourlist[skipCols:]

    hs = R.THStack("hs","")
    hist_count=0
    legend_hists=[]

    for i, hist in enumerate(hists):
        if norm_hists: hist.Scale(1.0/hist.Integral(0, hist.GetNbinsX()+1))
        if norm_bins: hist.Scale(1.0,"width")
        if scale_factors and len(scale_factors) == len (hists): hist.Scale(scale_factors[i])
        h = hist.Clone()
        objects.append(h)
        h.SetFillColor(0)
        h.SetLineWidth(3)
        h.SetLineColor(colourlist[hist_count])
        h.SetMarkerColor(colourlist[hist_count])
        h.SetMarkerSize(0)
        hs.Add(h)
        hist_count+=1
        o=h.Clone()
        objects.append(o)
        legend_hists.append(o)
        
    c1 = R.TCanvas()
    c1.cd()
    
    if ratio:
        pads=TwoPadSplit(0.29,0.01,0.01)
    else:
        pads=OnePad()
    pads[0].cd()
    
    if(log_y): pads[0].SetLogy(1)
    if(log_x): pads[0].SetLogx(1)
    if custom_x_range:
        if x_axis_max > hists[0].GetXaxis().GetXmax(): x_axis_max = hists[0].GetXaxis().GetXmax()
    if ratio:
        if(log_x): pads[1].SetLogx(1)
        axish = createAxisHists(2,hists[0],hists[0].GetXaxis().GetXmin(),hists[0].GetXaxis().GetXmax()-0.01)
        axish[1].GetXaxis().SetTitle(x_title)
        axish[1].GetXaxis().SetLabelSize(0.03)
        axish[1].GetYaxis().SetNdivisions(4)
        axish[1].GetYaxis().SetTitle("Ratio")
        axish[1].GetYaxis().SetTitleOffset(1.6)
        axish[1].GetYaxis().SetTitleSize(0.04)
        axish[1].GetYaxis().SetLabelSize(0.03)
    
        axish[0].GetXaxis().SetTitleSize(0)
        axish[0].GetXaxis().SetLabelSize(0)
        if custom_x_range:
          axish[0].GetXaxis().SetRangeUser(x_axis_min,x_axis_max-0.01)
          axish[1].GetXaxis().SetRangeUser(x_axis_min,x_axis_max-0.01)
        if custom_y_range:
          axish[0].GetYaxis().SetRangeUser(y_axis_min,y_axis_max)
          axish[1].GetYaxis().SetRangeUser(y_axis_min,y_axis_max)
    else:
        axish = createAxisHists(1,hists[0],hists[0].GetXaxis().GetXmin(),hists[0].GetXaxis().GetXmax()-0.005)
        axish[0].GetXaxis().SetTitle(x_title)
        if widePlot: 
            axish[0].GetXaxis().SetTitleSize(0.05)
            axish[0].GetXaxis().SetTitleOffset(0.95)
            axish[0].GetXaxis().SetLabelSize(0.04)
        else: 
            axish[0].GetXaxis().SetTitleSize(0.04)
            axish[0].GetXaxis().SetLabelSize(0.03)
        if custom_x_range:
          axish[0].GetXaxis().SetRangeUser(x_axis_min,x_axis_max-0.01)
        if custom_y_range:                                                                
          axish[0].GetYaxis().SetRangeUser(y_axis_min,y_axis_max)
    axish[0].GetYaxis().SetTitle(y_title)
    if widePlot: 
      axish[0].GetYaxis().SetTitleOffset(0.9)
      axish[0].GetYaxis().SetTitleSize(0.05)
      axish[0].GetYaxis().SetLabelSize(0.04)          
    else:      
      axish[0].GetYaxis().SetTitleOffset(1.6)
      axish[0].GetYaxis().SetTitleSize(0.04)
      axish[0].GetYaxis().SetLabelSize(0.03)

    axish[0].SetLineStyle(2)
    axish[0].SetLineColor(R.kBlack)

    hs.Draw("nostack same")

    extra_pad=max(0.2,extra_pad)
    if not custom_y_range:
        if(log_y):
            if hs.GetMinimum("nostack") >0: axish[0].SetMinimum(hs.GetMinimum("nostack"))
            else: axish[0].SetMinimum(0.0009)
            axish[0].SetMaximum(10**((1+extra_pad)*(math.log10(1.1*hs.GetMaximum("nostack") - math.log10(axish[0].GetMinimum())))))
        else:
            maxi=1.1*(1+extra_pad)*hs.GetMaximum("nostack")
            mini = None
            maxi = None
            for h in hists:
              if h is None: continue
              for i in range(1,h.GetNbinsX()+1): 
                lo = h.GetBinContent(i)#-h.GetBinError(i) 
                hi = h.GetBinContent(i)#+h.GetBinError(i) 
                if mini is None:
                  mini = min(lo,hi) 
                  maxi = max(lo,hi) 
                else:
                  mini = min(mini,lo,hi) 
                  maxi = max(maxi,lo,hi) 
            mini= -abs(mini)*(1.+extra_pad)
            axish[0].SetMinimum(mini)
            maxi*=(1.+extra_pad)
            axish[0].SetMaximum(maxi)
    axish[0].Draw()

    if IncErrors: hs.Draw("nostack hist e same")
    else: hs.Draw("nostack hist same")
    axish[0].Draw("axissame")
    
    #Setup legend
    tot = len(hists)
    #if tot < 4 and False: legend = PositionedLegend(0.20,0.3,3,0.05)
    #else: legend = PositionedLegend(0.27,0.3,3,0.04)
    leg_pos=3
    if lowerLeg: leg_pos = 6
    if wideLeg: legend = PositionedLegend(0.33,0.3,leg_pos,0.04)
    #else: legend = PositionedLegend(0.27,0.3,leg_pos,0.04)
    else: legend = PositionedLegend(0.2,0.3,leg_pos,0.04)
    #max_len = max([len(x) for x in legend_titles])
    #if max_len>20: legend = PositionedLegend(0.37,0.3,leg_pos,0.02)

    legend.SetFillStyle(0)
    legend.SetTextFont(42)
    if widePlot: legend.SetTextSize(0.05)
    else: legend.SetTextSize(0.032)
    legend.SetFillColor(0)
    

    for legi,hist in enumerate(legend_hists):
        legend.AddEntry(hist,legend_titles[legi],"l")
    col_count=1
    for i in range (legi+1,len(legend_titles)):
        legend.AddEntry(hist,legend_titles[i],"")#.SetTextColor(colourlist[col_count])
        col_count+=1 
   
    legend.Draw()
 
    DrawTitle(pads[0], title, 3)
    
    latex2 = R.TLatex()
    latex2.SetNDC()
    latex2.SetTextAngle(0)
    latex2.SetTextColor(R.kBlack)
    latex2.SetTextSize(0.028)
    #latex2.DrawLatex(0.145,0.955,label)
    DrawTitle(pads[0], label, 1)
    
    #Add ratio plot if required
    if ratio:
        ratio_hs = R.THStack("ratio_hs","")
        hist_count=0
        pads[1].cd()
        pads[1].SetGrid(0,1)
        axish[1].Draw("axis")
        axish[1].SetMinimum(float(ratio_range.split(',')[0]))
        axish[1].SetMaximum(float(ratio_range.split(',')[1]))
        div_hist = hists[0].Clone()
        objects.append(div_hist)
        
        for i in range(0,div_hist.GetNbinsX()+2): div_hist.SetBinError(i,0)
        first_hist=True
        for hist in hists:
            h = hist.Clone()
            objects.append(h)

            h.SetFillColor(0)
            h.SetLineWidth(3)
            h.SetLineColor(colourlist[hist_count])
            h.SetMarkerColor(colourlist[hist_count])
            h.SetMarkerSize(0)

            for i in range(1,div_hist.GetNbinsX()+1): div_hist.SetBinError(i,0.0)
            h.Divide(div_hist)
            #if first_hist and not IncErrors:
            #    for i in range(1,h.GetNbinsX()+1): h.SetBinError(i,0.00001)
            #    first_hist=False
            o = h.Clone()
            objects.append(o)
            ratio_hs.Add(o)
            hist_count+=1
        ratio_hs.Draw("nostack e hist same")  
        pads[1].RedrawAxis("G")
    pads[0].cd()
    pads[0].GetFrame().Draw()
    pads[0].RedrawAxis()
   
    c1.SaveAs(plot_name+'.pdf')
    c1.Close()

