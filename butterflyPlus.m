function butterflyPlus(data, timevec, chan_names,chan_locs, f)

% butterfly(data, timevec, chan_names,chan_locs)
% plot a buttefrly plot (all channels together) + the global field power
% (stndard deviation across channels). Clicking the right button on a line
% will display its number and its name if 'chan_names' are provided)
%
% data  channels x time array
% timevec  time vector (length should equal the number of columns as data)
% chan_names (optional) - an array of channel labels
% chan_locs (optional) - a file name of channel locations (same order as
%                        the channels in the data matrix; formatted for topoplot. If topoplot (from
%                        EEGlab) is not available, this option will not work). 
%
% Leon Deouell HUJI 14/3/2016
% Leon Deouell corrected such that the data is always read from the figure
% userdata so that it reflects the chnges done by rereferencing
% Leon Deouell added the option to add the data, time_vec and chan_names as
%              taken from an EEGlab structure. For example butterflyPlus(EEG.data, EEG.times,EEG.chanlocs,'head33.locs')
%              where EEG is the current dataset in the EEGlab workspace


if ~exist('f', 'var')
    try
        f = ERPfigure;
    catch
        f = figure;
    end
end

% reref = uimenu('Tag','reref','Label','Reference','callback','set(findobj(gcf,''tag'',''reref_list''),''visible'',''on'')');
% reref = uimenu('Tag','reref','Label','Reference','callback',{@toggle_visibility, get(hObject,'userdata')});
reref = uimenu('Tag','reref','Label','Reference','callback',@toggle_visibility);

if ~exist('timevec','var')
    timevec = 1: size(data,2);
elseif length(timevec) ~= size(data,2)
    error ('timevec doesn''t match the length of the data')
end
input.data = data;
input.current = data;
input.nchans = size(data,1);
input.timevec = timevec; 

%This is in case we use an EEGLAB structure. built on raw EEG dataset
if exist('chan_names','var') && ~iscell(chan_names) && isstruct(chan_names)
    if ~isfield(chan_names, 'labels')
        error('no labels field found')
    else
        nchans = length(chan_names);
        for ii = 1:nchans, ChanNames{ii} = chan_names(ii).labels; end 
        chan_names = ChanNames;
        clear ChanNames
    end
end
    

if exist('chan_names','var') && ~isempty(chan_names)
    if size(chan_names,1) < size(chan_names,2), chan_names = chan_names'; end
    if size(chan_names,2) ~= 1, error('channel names should have only 1 column of cells'), end
    input.chan_names = [chan_names;{'COMMON';'ORIGINAL'} ];
    reref_list = uicontrol(f,'style','popup','string',input.chan_names,'value',length(input.chan_names),'tag','reref_list','units','normalized','position',[0.2    0.95    0.1    0.02], 'visible','off','callback',@Change_reference);
    set(reref,'userdata',reref_list)
end

if exist('chan_locs','var') && ~isempty(chan_locs)
    input.chan_locs = chan_locs;
end
% set(f,'userdata', input)
chan2plot = 1:size(data, 1);
p = zeros(size(data, 1), 1);
for i = chan2plot
    hcmenu = uicontextmenu;
    hold on
    if exist('colorlist','var') %future feature
        plotcolor = colorolist(i,:);
    else
        plotcolor = [.8 .8 .8];
    end
    p(i) = plot(timevec, data(i,:),'color',[.8 .8 .8],'linewidth',1);
    if exist('chan_names','var') && ~isempty(chan_names)
        try
            label = [num2str(i) ': ' chan_names{i}];%if it is a cell array
        catch
            label = [num2str(i) ': ' chan_names(i,:)]; %if it is a char array
         
        end
    else
        label = num2str(i);
    end
        
    uimenu(hcmenu, 'Label',label);
    set(p(i), 'uicontextmenu', hcmenu);
    set(p(i),'buttondownfcn','set(gco,''color'', abs(get(gco,''color'')-[.8 .8 .8])), bring2front(gco, gca);');
end
axis tight
input.lines = p;
hcmenu = uicontextmenu;
GFP = std(data, 0, 1);
GFPplot = plot(timevec, GFP ,'color','b','linewidth',2);
uimenu(hcmenu, 'Label','GFP');
set(GFPplot, 'uicontextmenu', hcmenu);
set(GFPplot,'buttondownfcn','set(gco,''color'', abs(get(gco,''color'')-[.8 .8 .8])), bring2front(gco, gca);');
input.GFPplot = GFPplot;

special_menu = uimenu(f,'Label','Topo');
uimenu(special_menu,'Label','timeline','callback',@setTimeline)

set(f,'userdata', input)
set(gca,'userdata', input)


function setTimeline(hFigure, callbackdata)
currentpointer = get(gcf,'pointer');
currentwbuf = get(gcf,'WindowButtonUpFcn');
set(gcf,'pointer','cross')
set(gcf,'WindowButtonUpFcn','set(gcf,''pointer'',''arrow'')')
waitfor(gcf,'pointer')

if ~strcmp(get(gco,'type'),'line')
    return
end
xdata = get(gco,'xdata');
ylim = get(gca,'ylim');
ymin = ylim(1);
C = get (gca, 'CurrentPoint');
x = C(1,1);
xi = find_nearest_time(x,xdata);
x = xdata(xi);


vlines = plot([x x], get(gca,'ylim'),'tag','timeline');
col = get(vlines,'color');
set(vlines,'ButtonDownFcn',@GetTimeLine)
set(gcf,'WindowButtonUpFcn','set(gcf,''WindowButtonMotionFcn'','''')')

tbox = moveableTextBox(num2str(round(x,2)),[.05 .05 .07 0.07]);
set(tbox,'tag','timeline_text','color',col)
set(gcf,'pointer',currentpointer)
% set the context menu to delete the lines
hcmenu = uicontextmenu;
set(vlines, 'uicontextmenu', hcmenu);
uimenu(hcmenu, 'Label','Delete lines','callback','l = get(gco,''userdata'');, delete([l{1},l{2}])');
uimenu(hcmenu,'Label','Show topography','callback',@showtopo)
uimenu(hcmenu,'Label','Show scatterplot','callback',@scatter)
uimenu(hcmenu,'Label','Show 1D topo','callback',@show1Dtopo)
set(vlines,'userdata',{vlines, tbox})

function showtopo(hObject, callbackdata)
handles = get(gco,'userdata');
lines = handles{1};
C = get(lines(1),'xdata');
x = C(1);
input = get(gcf,'userdata');
xi = find(input.timevec == x);
if isempty(xi)
    error('couldn''t find xi in the time vector')
end
f = ERPfigure;
topoplot(input.current(:,xi),input.chan_locs,'electrodes','on','shading','interp','style','both','conv','on','emarker',{'.','k',32,2});
colormap(parula)
s = findobj(gcf,'type','surface'); % this is the colored surface
set(s,'buttondownfcn',['if strcmp(get(gco,''visible''),''on''),set(gco,''visible'',''off''), else, set(gco,''visible'',''on''), end'])
%         
title([ num2str(x) ])

function scatter(hObject, callbackdata)
handles = get(gco,'userdata');
lines = handles{1};
C = get(lines(1),'xdata');
x = C(1);
input = get(gcf,'userdata');
xi = find(input.timevec == x);
if isempty(xi)
    error('couldn''t find xi in the time vector')
end
f = ERPfigure;
toposcatter(input.current(:,xi),input.chan_locs);
title([ num2str(x) ])

function show1Dtopo(hObject, callbackdata)
handles = get(gco,'userdata');
lines = handles{1};
C = get(lines(1),'xdata');
x = C(1);
input = get(gcf,'userdata');
xi = find(input.timevec == x);
if isempty(xi)
    error('couldn''t find xi in the time vector')
end
f = ERPfigure;
plot(input.current(:,xi));
hold on
plot(input.current(:,xi),'or');
set(gca,'XTick',1:input.nchans, 'xticklabels',input.chan_names(1:input.nchans))
title([ num2str(x) ])

function GetTimeLine(hObject, callbackdata)
% set(gcf,'WindowButtonMotionFcn',['lines = get(gco,''userdata'');,moveline(lines)'])
set(gcf,'WindowButtonMotionFcn',{@MoveTimeLine,get(gco,'userdata')})

function MoveTimeLine(hObject, callbackdata,handles)
%lines is an array - the first element is the line handles and the second
%is the related textbox
input = get(gcf,'userdata');
C = get (gca, 'CurrentPoint');
xi = find_nearest_time(C(1,1),input.timevec);
x = input.timevec(xi);
lines = handles{1};
t = handles{2};
set(lines,'xdata', [x x])
%t = findall(gcf,'tag','timeline_text');
% pos = get(t(1),'position');
% set(t,'string',num2str(x), 'position',[round(x,2), pos(2), pos(3)])
set(t,'string',num2str(x))

function tbox = moveableTextBox(string, position)

%create text box in the figure that can be moved
if ~exist('position','var')
    position = [.05 .05 .07 .07]
end
% I use annotation below and not just text because text needs an axes and
% annotation does not (actually it does, but the annotation function
% creates one and hides it in the back ground apprently so it saves some
% lines of code here)
tbox = annotation('textbox','string', string, 'LineStyle','none','tag','timeline_text');
set(tbox, 'ButtonDownFcn',@moveObjwithPointer)
set(tbox,'position',position)


function moveObjwithPointer(hObject, callbackdata)
% move the current object to where the pointer is
curWinMotionFcn = get(gcf,'WindowButtonMotionFcn'); %store for later
curWinButtonUpFcn = get(gcf,'WindowButtonUpFcn'); %store for later
curPosition = get(hObject,'position');
set(gcf,'WindowButtonMotionFcn',@setnewpos)
set(gcf,'WindowButtonUpFcn',{@restoreFcns, curWinMotionFcn, curWinButtonUpFcn})

function restoreFcns(hObject, callbackdata, motionfcn,upfcn )
set(gcf,'WindowButtonMotionFcn',motionfcn)
set(gcf,'WindowButtonUpFcn',upfcn)

function newposition = setnewpos(hObject, callbackdata)
set(gco,'units',get(gcf,'units')) %make sure we speak the same language of units
boxposition = get(gco,'position');
newposition = [get(gcf,'CurrentPoint')-boxposition(3:4)/2 boxposition(3:4)]; % keep the size as before. We only want to move
% newposition = get(gcf,'CurrentPoint') ; % keep the size as before. We only want to move
set(gco,'position',newposition)

function Change_reference(hObject, callbackdata)
items = get(gco,'String');
chan_num_selected = get(gco,'Value');
chan_name_selected = items{chan_num_selected};
% display(chan_name_selected);
input = get(gcf,'userdata');
if strcmp(chan_name_selected, 'COMMON')
    input.current = bsxfun(@minus,input.data, mean(input.data));
elseif strcmp(chan_name_selected, 'ORIGINAL')
    input.current = input.data;
else
    input.current = bsxfun(@minus,input.data, input.data(chan_num_selected,:));
end
for i = 1:length(input.lines)
    set(input.lines(i),'ydata',input.current(i,:))
end
GFP = std(input.current, 0, 1);
set (input.GFPplot, 'ydata', GFP);
set(gcf,'userdata',input)

function toggle_visibility(hObject, callbackdata)
% keyboard
options = {'off','on'};
h = get(hObject,'userdata');
t = strcmp(get(h,'visible'),'off');
set(h, 'visible', options{t+1});





