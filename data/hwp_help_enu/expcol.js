var L_COLLAPSE_TEXT="Hide All";
var L_EXPAND_TEXT="Show All";

function ChangeFileName(fullpath, filename)
{
	var lastSlash = Math.max(fullpath.lastIndexOf('/'), fullpath.lastIndexOf('\\'));
	
	if (lastSlash > 0) {
		lastSlash = lastSlash + 1;
	}
	
	fullpath = fullpath.substr(0, lastSlash) + filename;
	return fullpath;
}

function ExpandAll()
{
	// Expand all 
	var iSpan;
	var iSpanSource;
	var oSpan;
	var oSpanChildren;
	var sCaption;
	var sAction;
	var sImage;
	
	// 액션 선택 및 그림 선택/변경
	sCaption=document.getElementById("ExpandAll").innerHTML;
	if (sCaption==L_EXPAND_TEXT)
		{
		sAction="expanded";
		sImage="icon_minus.gif";
		document.getElementById("ExpandAll").innerHTML=L_COLLAPSE_TEXT;
		}
	else
		{
		sAction="collapsed";
		sImage="icon_plus.gif";
		document.getElementById("ExpandAll").innerHTML=L_EXPAND_TEXT;
		}

	/*
	 * 2012-11-05 윤종현
	 * 아래 for 구문은 IE 로컬 파일로 실행될 경우 
	 * 교차스크립팅 방지를 위해 getElementsByTagName이 동작하지 않을 수 있다.
	 */
	// 모든 div를 열기	
	for (iSpan=0; iSpan < document.getElementsByTagName("DIV").length; iSpan++)
		{	
			oSpan=document.getElementsByTagName("DIV").item(iSpan);
			if (oSpan.id=="expcol")
			{
				document.getElementsByTagName("DIV").item(iSpan).className=sAction;
				// 아이콘 변경
				// IE : previousSibling을 탐색하여 이전 SiblingElement를 찾음
				if (oSpan.previousElementSibling) {
					oSpanChildren = oSpan.previousElementSibling.getElementsByTagName("IMG");
				} else {
					var elementNode = oSpan.previousSibling;
					while (elementNode.nodeType != 1 /*Node.ELEMENT_NODE*/) {
						elementNode = elementNode.previousSibling;
					}
					oSpanChildren = elementNode.getElementsByTagName("IMG");
				}

				for (var i=0; i<oSpanChildren.length; i++)
				{
					if (oSpanChildren[i].src.indexOf("icon_minus.gif") != -1 || oSpanChildren[i].src.indexOf("icon_plus.gif") != -1)
					{
						// 변경할 대상이 맞는 경우
						oSpanChildren[i].src = ChangeFileName(oSpanChildren[i].src, sImage);
						break;
					}
				}
			}			
			
		}


	// 텍스트 바꾸기
	
	document.getElementById("ExpandAll").className="DropDown";

	// 팝업 처리
	var cLinks = document.getElementsByTagName("A");
	var iNumLinks = cLinks.length;

  	for (var i=0;i<iNumLinks;i++)
  	{
		cLink=document.getElementsByTagName("A").item(i);
		
		switch (sAction)
		{
			case 'expanded':
				if ((cLink.className=="glossary" || cLink.className=="Glossary") && cLink.getAttribute("State")!="On")
				{
					cLinks[i].click();
				}
				else if (cLink.className=="HidePopUp")
				{
					cLinks[i].click();
				}
				
				break;
			case 'collapsed':
				if ((cLink.className=="glossary" || cLink.className=="Glossary") && cLink.getAttribute("State")=="On")
				{
					cLinks[i].click();
				}
				else if (cLink.className=="HidePopUp")
				{
					cLinks[i].click();
				}
				break;
		}
  	}
}

/* 
 * 2012-11-05 윤종현
 * Gecko 계열 브라우저는 이벤트를 이벤트 핸들러에게 인수로 전달하기 때문에
 * 반드시 event 관련 인수를 넘겨줘야만 하기 때문에, FF에서는 window.event를 정의하여 등록해둔다.
 * http://p2b.jp/index.php?UID=1149066600 
 */
if (navigator.userAgent.indexOf('Firefox') >= 0) {
	(function(){
		var events = ["mousedown","mouseover","mouseout","mousemove","mousedrag","click","dblclick"];
		for (var i = 0; i < events.length; i++) {
			window.addEventListener(events[i], function(e) {
				window.event = e;
			}, true);
		}
	}());
};

function expdiv()
{
	event.returnValue=0;
	// 클릭한 a href target을 찾는다.
	// 기본 : window.event.srcElement의 부모 a 태그, FF : window.event.target
	var elementA = event.srcElement ? checkParent(event.srcElement,"A") : event.target;
	var elementIMG = null;
	var elementDIV = null;
	if (null != elementA)
	{
		var elementIMGs = elementA.getElementsByTagName("IMG");
		for (var i=0; i<elementIMGs.length; i++)
		{
			if (elementIMGs[i].src.indexOf("icon_minus.gif") != -1 || elementIMGs[i].src.indexOf("icon_plus.gif") != -1)
			{
				// 변경할 대상이 맞는 경우
				elementIMG = elementIMGs[i];
				break;
			}
		}
		// a 이후에 나오는 expcol DIV를 찾는다.
		var isFound = false;
		for (var fDIV = elementA;
			fDIV != null;
			fDIV = fDIV.parentElement) 
		{
			var fDIVsibling = fDIV;
			while (true)
			{
				if (fDIVsibling.tagName == "DIV" && fDIVsibling.id == "expcol") 
				{
					elementDIV = fDIVsibling;
					isFound = true;
					break;
				}
				if (fDIVsibling.nextSibling == null) break;
				fDIVsibling = fDIVsibling.nextSibling;
			};
			if (isFound == true) break;
		}
		if (elementIMG == null || elementDIV == null) return;

		if(elementDIV.className=="collapsed")
		{
			elementDIV.className="expanded";
			elementIMG.src = ChangeFileName(elementIMG.src, "icon_minus.gif");
		}
		else if(elementDIV.className=="expanded")
		{
			elementDIV.className="collapsed";
			elementIMG.src = ChangeFileName(elementIMG.src, "icon_plus.gif");
		}
		else
		{
			return;
		}
		event.cancelBubble = true;
	}
}


function checkParent(src,dest)
{
	//Search for a specific parent of the current element.
	while(src !=null)
	{
		if(src.tagName == dest)
		{
			return src;
		}
		src = src.parentElement;
	}
	return null;
}
