package org.lungen.deeplearning.iterator;

import org.junit.Assert;
import org.junit.Test;

import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static org.lungen.deeplearning.iterator.CSVParser.quote;


/**
 * TestCSVParserBugzilla
 *
 * @author lungen.tech@gmail.com
 */
public class TestCSVParserBugzilla {


    @Test
    public void testProcessed() {
        String dir = "C:/DATA/Projects/DataSets/Bugzilla";
        String filePath = dir + "/bugzilla-processed.csv";
        CSVParser parser = new CSVParser(new File(filePath), true);
        List<List<String>> parsedLines = parser.getParsedLines();
        Assert.assertEquals(40980, parsedLines.size());
    }

    @Test
    public void testPreProcess() {
        String dir = "C:/DATA/Projects/DataSets/Bugzilla";
        String filePath = dir + "/bugzilla-test1.csv";
        CSVParserBugzilla parser = new CSVParserBugzilla(new File(filePath), true);
        parser.process();
        List<List<String>> parsedLines = parser.getParsedLines();
        Assert.assertEquals(15, parsedLines.size());
        for (int i = 0; i < parsedLines.size(); i++) {
            Assert.assertEquals("" + (10565 + i),
                    parsedLines.get(i).get(parser.headers.get("id")));
        }
    }

    @Test
    public void testParseSummaryQuotes() {
        String dir = "C:/DATA/Projects/DataSets/Bugzilla";
        String filePath = dir + "/bugzilla-test2.csv";
        CSVParserBugzilla parser = new CSVParserBugzilla(new File(filePath), true);
        List<List<String>> parsedLines = parser.getParsedLines();

        List<String> line0 = parsedLines.get(0);
        Assert.assertEquals(quote("It's possible have some identical transitions from one block (such \"\"If\"\", \"\"Loop\"\").This is hidden for Errors View."),
                line0.get(parser.headers.get("summary")));
        Assert.assertEquals(quote("{'is_private': False, 'count': 0, 'attachment_id': None, 'creator': 'sgritsenko@jnetx.ru', 'time': '2003-11-10T10:56:58Z', 'bug_id': 149, 'tags': [], 'text': 'It\\'s possible have some identical transitions from one block.\\nTry next actions:\\n1. Create diagram.\\n2. Set \"\"Listen Call\"\" block -> \"\"If\"\" block.\\n3. From \"\"If\"\" block pass 2 transitions (named equally) to \"\"Log\"\" blocks.\\nAs result, this effect doesn\\'t show in \"\"Errors View\"\" and deployment accepted \\naltough this will be error in runtime.\\n\\nThe same with block \"\"Loop\"\" (Tag \"\"Next\"\" be repeated).\\n\\n\\n Tag : HEAD (10.11.2003) TSS.', 'id': 263, 'creation_time': '2003-11-10T10:56:58Z'}"),
                line0.get(parser.headers.get("description")));

        List<String> line1 = parsedLines.get(1);
        Assert.assertEquals(quote("Diagram which consist from \"\"Basic\"\" category components cannot be deployed without CC libraries."),
                line1.get(parser.headers.get("summary")));
        Assert.assertEquals(quote("{'is_private': False, 'count': 0, 'attachment_id': None, 'creator': 'sgritsenko@jnetx.ru', 'time': '2003-11-10T11:26:22Z', 'bug_id': 150, 'tags': [], 'text': 'I create diagram that consist all components from category \"\"Basic\"\".\\nI haven\\'t CC libraries in \"\"libs\"\" directory of appdesigner plugin.\\nAs result I have errors during deploy procedure. My error output are :\\n\\nD:eclipsepluginscom.jnetx.appdesigner.eclipse\\x08uild-\\noutcomjnetxgeneratedapplication7application7Alarm1.java:5: package \\njavax.jain.services.jcc does not exist\\nimport javax.jain.services.jcc.*;\\n^\\nD:eclipsepluginscom.jnetx.appdesigner.eclipse\\x08uild-\\noutcomjnetxgeneratedapplication7Application7Resource.java:20: package \\ncom.jnetx.slee.rb.cc does not exist\\nimport com.jnetx.slee.rb.cc.*;\\n^\\nD:eclipsepluginscom.jnetx.appdesigner.eclipse\\x08uild-\\noutcomjnetxgeneratedapplication7Application7Resource.java:21: package \\ncom.jnetx.slee.rb.cc.events does not exist\\nimport com.jnetx.slee.rb.cc.events.*;\\n^\\nD:eclipsepluginscom.jnetx.appdesigner.eclipse\\x08uild-\\noutcomjnetxgeneratedapplication7Application7Resource.java:23: package \\ncom.jnetx.scf.cc does not exist\\nimport com.jnetx.scf.cc.*;\\n^\\nD:eclipsepluginscom.jnetx.appdesigner.eclipse\\x08uild-\\noutcomjnetxgeneratedapplication7Application7Resource.java:24: package \\ncom.jnetx.scf does not exist\\nimport com.jnetx.scf.AddressRange;\\n                     ^\\nD:eclipsepluginscom.jnetx.appdesigner.eclipse\\x08uild-\\noutcomjnetxgeneratedapplication7Application7Resource.java:25: package \\norg.csapi.cc does not exist\\nimport org.csapi.cc.TpCallEventType;\\n                    ^\\nD:eclipsepluginscom.jnetx.appdesigner.eclipse\\x08uild-\\noutcomjnetxgeneratedapplication7Application7Resource.java:26: package \\norg.csapi.cc does not exist\\nimport org.csapi.cc.TpCallMonitorMode;\\n                    ^\\nD:eclipsepluginscom.jnetx.appdesigner.eclipse\\x08uild-\\noutcomjnetxgeneratedapplication7Application7Resource.java:27: package \\norg.csapi does not exist\\nimport org.csapi.TpAddressPlan;\\n                 ^\\n8 errors\\n\\nTag : HEAD of TSS (10.11.2003)', 'id': 265, 'creation_time': '2003-11-10T11:26:22Z'}"),
                line1.get(parser.headers.get("description")));

        List<String> line2 = parsedLines.get(2);
        Assert.assertEquals(quote("SSC sends this alarm : \"\"Provisioning problem: the terminating subscriber '353857279952' is not found in the database!\"\""),
                line2.get(parser.headers.get("summary")));
        Assert.assertEquals(quote("{'is_private': False, 'count': 0, 'attachment_id': None, 'creator': 'aavvakumov@amdocs.com', 'time': '2003-11-10T16:48:47Z', 'bug_id': 163, 'tags': [], 'text': '20030829_1945_MeteorJune\\n\\n\\nSSC sends this alarm : \"\"Provisioning problem: the terminating \\nsubscriber \\'353857279952\\' is not found in the database!\"\"\\n\\nIt happends when someone dials a number that looks like a valid Meteor number, \\nbut the subsciber with this number dosn\\'t exist in the database.\\nIt means that system administarator receives this alarm each time a user dials \\nincorrect number. \\n\\nThis is not a probisioning problem!!!', 'id': 289, 'creation_time': '2003-11-10T16:48:47Z'}"),
                line2.get(parser.headers.get("description")));

        List<String> line3 = parsedLines.get(3);
        Assert.assertEquals("tsi server doesnot read input buffer correctly",
                line3.get(parser.headers.get("summary")));
        Assert.assertEquals(quote("{'is_private': False, 'count': 0, 'attachment_id': None, 'creator': 'sytsko@amdocs.com', 'time': '2003-12-11T15:08:58Z', 'bug_id': 365, 'tags': [], 'text': '2003-12-10 20:36:04,007 ERROR tsi                : InvalidPDUException caught \\nexception receiving PDU from SMSC.\\nPDU debug string: TransactionPDU: , Operation: 0, AccountIndicator: 1, \\nlAccountIndicator: 1, TransactionID: \\x01\\x0110000031?, SourceAddrTON: 1, \\nSourceAddrNPI: 1, SourceAddress: NOT PRESENT, DestAddrTON: 0, DestAddrNPI: 0, \\nDestAddress: NOT PRESENTcom.jnetx.slee.resources.tsi.axl.pdu.\\nTransactionPDU@24c22b', 'id': 840, 'creation_time': '2003-12-11T15:08:58Z'}"),
                line3.get(parser.headers.get("description")));

        List<String> line4 = parsedLines.get(4);
        Assert.assertEquals(quote("[KT][OCA][auto] unexpected element (uri:\"\"\"\", local:\"\"responseText\"\") exception for l9BigiRecharge (when error from TC was received)"),
                line4.get(parser.headers.get("summary")));
        Assert.assertEquals(quote("{'is_private': False, 'count': 0, 'attachment_id': None, 'creator': 'maximo@amdocs.com', 'time': '2016-12-14T12:43:53Z', 'bug_id': 38302, 'tags': [], 'text': 'Build Name: 20161214_133324_KT\\nsteps:\\n\\n1. send l9BigiRecharge soap request\\n2. reply by some error from TC emulator\\n\\nexpected:\\nfollowing fields are expected in soap reply (according to setCreditLimitResponse \\ntype in KT_OCA_TC_Ro.wsdl):\\n<responseCode>\\n<responseTest>\\n<responseDet>\\n\\nactual:\\nunexpected element (uri:\"\"\"\", local:\"\"responseText\"\") received in reply from OCA:\\n\\n\\n<soapenv:Envelope xmlns:soapenv=\"\"http://schemas.xmlsoap.org/soap/envelope/\"\" \\nxmlns:wsc=\"\"http://wsc.kt.service.wsdl.amdocs.com/\"\">\\n   <soapenv:Header/>\\n   <soapenv:Body>\\n      <wsc:l9BigiRechargeResponse>\\n         <return>\\n            <responseCode>320</responseCode>\\n            <!--Optional:-->\\n            <responseText>Internal TC error code</responseText>\\n         </return>\\n      </wsc:l9BigiRechargeResponse>\\n   </soapenv:Body>\\n</soapenv:Envelope>', 'id': 316341, 'creation_time': '2016-12-14T12:43:53Z'}"),
                line4.get(parser.headers.get("description")));

    }

    @Test
    public void testParseOneLine() {
        Map<String, Integer> indexMap = new HashMap<>();
        String headerStr = "id,status,resolution,severity,priority,creation_time,creator,assigned_to,component,resolution_dates,summary,description,is_open,creator_detail,assigned_to_detail,blocks,last_change_time,is_cc_accessible,keywords,cf_ext_bug_reference,cc,url,groups,see_also,whiteboard,qa_contact,cf_current_status,depends_on,dupe_of,cf_expected_fix_date,estimated_time,cf_build_name,remaining_time,cf_testlink_tc,qa_contact_detail,update_token,classification,alias,op_sys,cf_status_whiteboard,cc_detail,cf_responsible,platform,cf_responsible_detail,flags,version,deadline,actual_time,is_creator_accessible,product,is_confirmed,target_milestone";
        String[] headers = headerStr.split(",");
        IntStream.range(0, headers.length).forEach(i -> indexMap.put(headers[i], i));
        Assert.assertEquals(52, indexMap.size());

        // 52 fields
        String id = "42367";
        String status = "RESOLVED";
        String resolution = "FIXED";
        String severity = "normal";
        String priority = "P3";
        String creationTime = "2018-11-01T10:25:43Z";
        String creator = "sdmitry@amdocs.com";
        String assignedTo = "istrelnikov@amdocs.com";
        String component = "anm-custom";
        String resolutionDates = "['2018-11-01T13:24:26Z']";
        String summary = "[ANM] [MAXIS] Record with DEFERRED status is not written to history in case of postponed Welcome message (FEAT-17197)";
        String description = "\"{'is_private': False, 'count': 0, 'attachment_id': 38970, 'creator': 'sdmitry@amdocs.com', 'time': '2018-11-01T10:25:43Z', 'bug_id': 42367, 'tags': [], 'text': 'Created attachment 38970\\nSlee log\\n\\nBuild Name: 20181101_030138_MAXIS\\nHistory record with DEFERRED status is not written to history in case of postponed Welcome message (FEAT-17197)\\n\\nFrom Slee log:\\n\\n2018-11-01 12:46:13,374 [ER-8] DEBUG tss-sm.tss-generated.TRIGGER_SOAP_V3 : [49]INCOMING SOAP:\\n<soap:Envelope xmlns:soap=\"\"http://schemas.xmlsoap.org/soap/envelope/\"\">\\n    <soap:Body>\\n        <ns3:pushRequest xmlns:ns2=\"\"http://amdocs.com/schema/service/asmfw/types/v2_0\"\"\\n                         xmlns:ns3=\"\"http://amdocs.com/wsdl/service/nm/triggering/v3_0\"\">\\n            <ns3:event xmlns:xsi=\"\"http://www.w3.org/2001/XMLSchema-instance\"\"\\n                       xmlns:ns5=\"\"http://amdocs.com/schema/service/nm/types/v3_0\"\" xsi:type=\"\"ns5:NotificationEventIdentifier\"\">\\n                <eventId>NJJNotification</eventId>\\n            </ns3:event>\\n            <ns3:target xmlns:xsi=\"\"http://www.w3.org/2001/XMLSchema-instance\"\"\\n                        xmlns:ns5=\"\"http://amdocs.com/schema/service/nm/types/v3_0\"\" xsi:type=\"\"ns5:NotificationTargetIdentifier\"\">\\n                <targetId>gr_17867</targetId>\\n            </ns3:target>\\n            <ns3:templateParameter>\\n                <name>NJJ day</name>\\n                <value xmlns:xsi=\"\"http://www.w3.org/2001/XMLSchema-instance\"\"\\n                       xmlns:ns5=\"\"http://amdocs.com/schema/service/nm/types/v3_0\"\" xsi:type=\"\"ns5:IntegerParameterValue\"\">\\n                    <integerValue>0</integerValue>\\n                </value>\\n            </ns3:templateParameter>\\n            <ns3:templateParameter>\\n                <name>NJJ notification number</name>\\n                <value xmlns:xsi=\"\"http://www.w3.org/2001/XMLSchema-instance\"\"\\n                       xmlns:ns5=\"\"http://amdocs.com/schema/service/nm/types/v3_0\"\" xsi:type=\"\"ns5:IntegerParameterValue\"\">\\n                    <integerValue>1</integerValue>\\n                </value>\\n            </ns3:templateParameter>\\n        </ns3:pushRequest>\\n    </soap:Body>\\n</soap:Envelope>\\n        2018-11-01 12:46:13,375 [ER-8] DEBUG tss-sm.tss-generated.TRIGGER_SOAP_V3 : [49]Operation name: PushNotification\\n        2018-11-01 12:46:13,376 [ER-8] DEBUG anm-nm-ra.NotificationGatewayResourceAdaptor : processing notification request: PushNotificationRequest{operatorId=null, operatorName=\\'null\\', sourceId=\\'null\\', event=NotificationEventIdentifier{eventId=\\'NJJNotification\\'}, callback=null, reportDelivery=null, expirationDate=null, deliveryDateTime=null, priority=null, notificationId=\\'null\\', targets=[NotificationTargetIdentifier{tag=\\'null\\', timeZone=\\'null\\', targetId=\\'gr_17867\\', idScope=\\'null\\', filterChannelTypeSet=\\'null\\', overrideSources=null}], templateParameters=[TemplateParameter{name=\\'NJJ day\\', value=\\'ParameterValue{value=\\'0\\', type=Integer, isArray=false}\\', sensitive=\\'false\\'}, TemplateParameter{name=\\'NJJ notification number\\', value=\\'ParameterValue{value=\\'1\\', type=Integer, isArray=false}\\', sensitive=\\'false\\'}], templateParametersMap={NJJ day=TemplateParameter{name=\\'NJJ day\\', value=\\'ParameterValue{value=\\'0\\', type=Integer, isArray=false}\\', sensitive=\\'false\\'}, NJJ notification number=TemplateParameter{name=\\'NJJ notification number\\', value=\\'ParameterValue{value=\\'1\\', type=Integer, isArray=false}\\', sensitive=\\'false\\'}}, isCorrect=true, parsingError=null, userName=\\'null\\'}\\n        2018-11-01 12:46:13,376 [ER-8] INFO  ngw.ra.policy.LocalShardingPolicy : LocalShardingPolicy created: currentNodeId=1\\n        2018-11-01 12:46:13,376 [ER-8] DEBUG GUIDGenerator      : GUID generated: 3231848749336657920 (Time: 2018.11.01 12:46:13.376; Node identifier:1; Inc:0)\\n        2018-11-01 12:46:13,376 [ER-8] DEBUG anm-dal-jdbc.OracleDataProviderDicImpl : findDefaultOperator(), got it from cache.\\n        2018-11-01 12:46:13,376 [ER-8] DEBUG anm-nm-ra.PushRequestParser : Expiration date not specified, use default 24h\\n        2018-11-01 12:46:13,381 [ER-8] DEBUG rule-engine-core.rule-peer : Condition for rule \"\"ANM preparty rule\"\", result = true\\n        2018-11-01 12:46:13,381 [ER-8] DEBUG ngw.ra.policy.LocalShardingPolicy : calculateNodeId(address=\\'null\\'): 1\\n        2018-11-01 12:46:13,381 [ER-8] DEBUG GUIDGenerator      : GUID generated: 3231848749347143680 (Time: 2018.11.01 12:46:13.381; Node identifier:1; Inc:0)\\n        2018-11-01 12:46:13,382 [ER-8] DEBUG anm.deferred.LockingDeferredProvider : persistDeferredRecord: rec=DeferredRecord{isPhantom=true, id=3231848749347143680, guardId=null, nodeId=1, status=PLANNED, notificationId=\\'anm#3231848749336657920\\', triggeredTime=Thu Nov 01 12:46:13 MSK 2018, retryCount=0, recordTime=null, nextAttemptTime=Thu Nov 01 12:48:13 MSK 2018, expirationTime=Fri Nov 02 12:46:13 MSK 2018, operatorId=0, sourceId=\\'null\\', retryPeriod=0, channelId=0, channelType=ChannelType{name=\\'UNKNOWN\\', code=0, title=\\'Unknown\\', deliveryReportSupported=false, addressRequired=false}, protocolName=\\'undefined\\', address=\\'null\\', deliveryIndex=0, targetTag=\\'null\\', targetExtId=\\'null\\', targetIdScope=\\'null\\', originalTargetId=\\'null\\', originalTargetIdScope=\\'null\\', eventType=\\'null\\', subject=\\'null\\', message=\\'null\\', contentType=\\'null\\', reportRequestForced=null, templateId=null, endpointTypeId=null, params=\\'null\\', deliveryHint=\\'null\\', reportRequested=false, isEventSensitive=false, enhancers=null, currentEnhancer=-1, enhancerRetryCount=1, priority=NORMAL, sourceAddress=\\'null\\', request=\\'{\"\"correct\"\":true,\"\"deliveryDateTime\"\":\"\"2018-11-01T12:48:13\"\",\"\"event\"\":{\"\"eventId\"\":\"\"NJJNotification\"\"},\"\"expirationDate\"\":\"\"2018-11-02T12:46:13\"\",\"\"notificationId\"\":\"\"anm#3231848749336657920\"\",\"\"targets\"\":[{\"\"targetId\"\":\"\"gr_17867\"\"}],\"\"templateParameters\"\":[{\"\"n\"\":\"\"NJJ day\"\",\"\"v\"\":{\"\"t\"\":3,\"\"v\"\":0}},{\"\"n\"\":\"\"NJJ notification number\"\",\"\"v\"\":{\"\"t\"\":3,\"\"v\"\":1}}]}\\'}\\n        2018-11-01 12:46:13,382 [ER-8] DEBUG anm.deferred.LockingDeferredProvider : persistDeferredRecord: writing using deferred-batch-executor\\n        2018-11-01 12:46:13,383 [ER-8] DEBUG tss-sm.tss-generated.TRIGGER_SOAP_V3 : [49]OUTGOING SOAP:\\n        <?xml version=\"\"1.0\"\"?><soap:Envelope xmlns:soap=\"\"http://schemas.xmlsoap.org/soap/envelope/\"\"\\n                                            xmlns:xsi=\"\"http://www.w3.org/2001/XMLSchema-instance\"\"\\n                                            xmlns:types=\"\"http://amdocs.com/schema/service/nm/types/v3_0\"\"\\n                                            xmlns:ns=\"\"http://amdocs.com/wsdl/service/nm/triggering/v3_0\"\"\\n                                            xmlns:asmfw=\"\"http://amdocs.com/schema/service/asmfw/types/v2_0\"\">\\n<soap:Body>\\n    <ns:pushResult>\\n        <ns:apiVersion>0</ns:apiVersion>\\n        <ns:event xsi:type=\"\"types:NotificationEventIdentifier\"\">\\n            <eventId><![CDATA[NJJNotification]]></eventId>\\n        </ns:event>\\n        <ns:notificationId><![CDATA[anm#3231848749336657920]]></ns:notificationId>\\n        <ns:\\n        <asmfw:result xmlns:asmfw=\"\"http://amdocs.com/schema/service/asmfw/types/v2_0\"\">true</asmfw:result>\\n    </ns:\\n    <ns:target xsi:type=\"\"types:NotificationTargetIdentifier\"\">\\n        <targetId><![CDATA[gr_17867]]></targetId>\\n    </ns:target>\\n</ns:pushResult>\\n</soap:Body>\\n        </soap:Envelope>', 'id': 341579, 'creation_time': '2018-11-01T10:25:43Z'}\"";
        String isOpen = "False";
        String creatorDetail = "\"{'email': 'sdmitry@amdocs.com', 'real_name': 'Dmitry Sobetsky', 'name': 'sdmitry@amdocs.com', 'id': 840}\"";
        String assignedToDetail = "\"{'email': 'istrelnikov@amdocs.com', 'real_name': 'Ivan A. Strelnikov', 'name': 'istrelnikov@amdocs.com', 'id': 257}\"";
        String blocks = "[]";
        String lastChangeTime = "2018-11-06T16:34:40Z";
        String isCCAccessible = "True";
        String keywords = "[]";
        String cfExtBugReference = "";
        String cc = "['istrelnikov@amdocs.com']";
        String url = "";
        String groups = "\"['Amdocs ASP Customers', 'jNETX']\"";
        String seeAlso = "[]";
        String whiteboard = "";
        String qaContact = "JnetXTestLeaders@amdocs.com";
        String cfCurrentStatus = "";
        String dependsOn = "[]";
        String dupeOf = "";
        String cfExpectedFixDate = "";
        String estimatedTime = "0";
        String cfBuildName = "20181101_030138_MAXIS";
        String remainingTime = "0";
        String cfTestlinkTc = "";
        String qaContactDetail = "\"{'email': 'JnetXTestLeaders@amdocs.com', 'real_name': 'QA mail list. Includes V-n-V TAC and Integration', 'name': 'JnetXTestLeaders@amdocs.com', 'id': 58}\"";
        String updateToken = "1541567346-iFpoPBCxQtlfoWiFtWD2jRZ6H0hPPkeWxYVebUxpuGk";
        String classification = "Unclassified";
        String alias = "[]";
        String opSys = "Linux";
        String cfStatusWhiteboard = "";
        String ccDetail = "\"[{'email': 'istrelnikov@amdocs.com', 'real_name': 'Ivan A. Strelnikov', 'name': 'istrelnikov@amdocs.com', 'id': 257}]\"";
        String cfResponsible = "";
        String platform = "Other";
        String cfResponsibleDetail = "\"{'email': '', 'real_name': '', 'name': '', 'id': 0}\"";
        String flags = "[]";
        String version = "unspecified";
        String deadline = "";
        String actualTime = "0";
        String isCreatorAccessible = "True";
        String product = "ASP";
        String isConfirmed = "True";
        String targetMilestone = "---";

        String line = id + "," +
                status + "," +
                resolution + "," +
                severity + "," +
                priority + "," +
                creationTime + "," +
                creator + "," +
                assignedTo + "," +
                component + "," +
                resolutionDates + "," +
                summary + "," +
                description + "," +
                isOpen + "," +
                creatorDetail + "," +
                assignedToDetail + "," +
                blocks + "," +
                lastChangeTime + "," +
                isCCAccessible + "," +
                keywords + "," +
                cfExtBugReference + "," +
                cc + "," +
                url + "," +
                groups + "," +
                seeAlso + "," +
                whiteboard + "," +
                qaContact + "," +
                cfCurrentStatus + "," +
                dependsOn + "," +
                dupeOf + "," +
                cfExpectedFixDate + "," +
                estimatedTime + "," +
                cfBuildName + "," +
                remainingTime + "," +
                cfTestlinkTc + "," +
                qaContactDetail + "," +
                updateToken + "," +
                classification + "," +
                alias + "," +
                opSys + "," +
                cfStatusWhiteboard + "," +
                ccDetail + "," +
                cfResponsible + "," +
                platform + "," +
                cfResponsibleDetail + "," +
                flags + "," +
                version + "," +
                deadline + "," +
                actualTime + "," +
                isCreatorAccessible + "," +
                product + "," +
                isConfirmed + "," +
                targetMilestone;

        List<String> values = CSVParser.parseCSVLine(line);
//        Assert.assertEquals(52, values.size());
        Assert.assertEquals(id, values.get(indexMap.get("id")));
        Assert.assertEquals(status, values.get(indexMap.get("status")));
        Assert.assertEquals(resolution, values.get(indexMap.get("resolution")));
        Assert.assertEquals(severity, values.get(indexMap.get("severity")));
        Assert.assertEquals(priority, values.get(indexMap.get("priority")));
        Assert.assertEquals(creationTime, values.get(indexMap.get("creation_time")));
        Assert.assertEquals(creator, values.get(indexMap.get("creator")));
        Assert.assertEquals(assignedTo, values.get(indexMap.get("assigned_to")));
        Assert.assertEquals(component, values.get(indexMap.get("component")));
        Assert.assertEquals(resolutionDates, values.get(indexMap.get("resolution_dates")));
        Assert.assertEquals(summary, values.get(indexMap.get("summary")));
        Assert.assertEquals(description, values.get(indexMap.get("description")));
        Assert.assertEquals(isOpen, values.get(indexMap.get("is_open")));
        Assert.assertEquals(creatorDetail, values.get(indexMap.get("creator_detail")));
        Assert.assertEquals(assignedToDetail, values.get(indexMap.get("assigned_to_detail")));
        Assert.assertEquals(blocks, values.get(indexMap.get("blocks")));
        Assert.assertEquals(lastChangeTime, values.get(indexMap.get("last_change_time")));
        Assert.assertEquals(isCCAccessible, values.get(indexMap.get("is_cc_accessible")));
        Assert.assertEquals(keywords, values.get(indexMap.get("keywords")));
        Assert.assertEquals(cfExtBugReference, values.get(indexMap.get("cf_ext_bug_reference")));
        Assert.assertEquals(cc, values.get(indexMap.get("cc")));
        Assert.assertEquals(url, values.get(indexMap.get("url")));
        Assert.assertEquals(groups, values.get(indexMap.get("groups")));
        Assert.assertEquals(seeAlso, values.get(indexMap.get("see_also")));
        Assert.assertEquals(whiteboard, values.get(indexMap.get("whiteboard")));
        Assert.assertEquals(qaContact, values.get(indexMap.get("qa_contact")));
        Assert.assertEquals(cfCurrentStatus, values.get(indexMap.get("cf_current_status")));
        Assert.assertEquals(dependsOn, values.get(indexMap.get("depends_on")));
        Assert.assertEquals(dupeOf, values.get(indexMap.get("dupe_of")));
        Assert.assertEquals(cfExpectedFixDate, values.get(indexMap.get("cf_expected_fix_date")));
        Assert.assertEquals(estimatedTime, values.get(indexMap.get("estimated_time")));
        Assert.assertEquals(cfBuildName, values.get(indexMap.get("cf_build_name")));
        Assert.assertEquals(remainingTime, values.get(indexMap.get("remaining_time")));
        Assert.assertEquals(cfTestlinkTc, values.get(indexMap.get("cf_testlink_tc")));
        Assert.assertEquals(qaContactDetail, values.get(indexMap.get("qa_contact_detail")));
        Assert.assertEquals(updateToken, values.get(indexMap.get("update_token")));
        Assert.assertEquals(classification, values.get(indexMap.get("classification")));
        Assert.assertEquals(alias, values.get(indexMap.get("alias")));
        Assert.assertEquals(opSys, values.get(indexMap.get("op_sys")));
        Assert.assertEquals(cfStatusWhiteboard, values.get(indexMap.get("cf_status_whiteboard")));
        Assert.assertEquals(ccDetail, values.get(indexMap.get("cc_detail")));
        Assert.assertEquals(cfResponsible, values.get(indexMap.get("cf_responsible")));
        Assert.assertEquals(platform, values.get(indexMap.get("platform")));
        Assert.assertEquals(cfResponsibleDetail, values.get(indexMap.get("cf_responsible_detail")));
        Assert.assertEquals(flags, values.get(indexMap.get("flags")));
        Assert.assertEquals(version, values.get(indexMap.get("version")));
        Assert.assertEquals(deadline, values.get(indexMap.get("deadline")));
        Assert.assertEquals(actualTime, values.get(indexMap.get("actual_time")));
        Assert.assertEquals(isCreatorAccessible, values.get(indexMap.get("is_creator_accessible")));
        Assert.assertEquals(product, values.get(indexMap.get("product")));
        Assert.assertEquals(isConfirmed, values.get(indexMap.get("is_confirmed")));
        Assert.assertEquals(targetMilestone, values.get(indexMap.get("target_milestone")));
    }

    @Test
    public void testParseFile() {
        String dir = "C:/DATA/Projects/DataSets/Bugzilla";
        String filePath = dir + "/bugzilla-test0.csv";
        CSVParserBugzilla parser = new CSVParserBugzilla(new File(filePath), true);
        parser.process();

        List<List<String>> parsedLines = parser.getParsedLines();
        Assert.assertEquals(9, parsedLines.size());

        List<String> parsed1 = parsedLines.get(0);
        Assert.assertEquals("42365", parsed1.get(parser.headers.get("id")));
        Assert.assertEquals("RESOLVED", parsed1.get(parser.headers.get("status")));
        Assert.assertEquals("FIXED", parsed1.get(parser.headers.get("resolution")));
        Assert.assertEquals("normal", parsed1.get(parser.headers.get("severity")));
        Assert.assertEquals("P3", parsed1.get(parser.headers.get("priority")));
        Assert.assertEquals("pznamenskii@amdocs.com", parsed1.get(parser.headers.get("creator")));
        Assert.assertEquals("olukyanov@amdocs.com", parsed1.get(parser.headers.get("assigned_to")));
        Assert.assertEquals("platform-scripting", parsed1.get(parser.headers.get("component")));
        Assert.assertEquals("[]", parsed1.get(parser.headers.get("blocks")));
        Assert.assertEquals("[]", parsed1.get(parser.headers.get("depends_on")));
        Assert.assertEquals("", parsed1.get(parser.headers.get("dupe_of")));
        Assert.assertEquals(quote("[XL][SCP] update failed due to an unknown reason\\nCreated attachment 38967\\nupdate log\\n\\nBuild Name: \\nbuild before: 20180911_111654_XLIN\\nnew build: 20181031_170240\\n\\nDuring the asp-deploy (after Full Upgrade via asp-asc-20181025_115313_XLIN.bin -U), an error pops up:\\n\\n[fatal]: Exception caught during execution of add command\\n\\nas the result:\\n- no backup in $ASP_HOME/backup\\n- no CDRs restored in slee/data\\n\\nplease check and let us know the cause and the way to fix it."), parsed1.get(parser.headers.get("description")));
        Assert.assertEquals("2018-10-31T18:08:34Z", parsed1.get(parser.headers.get("creation_time")));
        Assert.assertEquals("2018-11-02T07:37:37Z", parsed1.get(parser.headers.get("resolution_time")));
        Assert.assertEquals("37", parsed1.get(parser.headers.get("resolution_duration_hours")));

        List<String> parsed9 = parsedLines.get(8);
        Assert.assertEquals("42383", parsed9.get(parser.headers.get("id")));
        Assert.assertEquals("RESOLVED", parsed9.get(parser.headers.get("status")));
        Assert.assertEquals("FIXED", parsed9.get(parser.headers.get("resolution")));
        Assert.assertEquals("normal", parsed9.get(parser.headers.get("severity")));
        Assert.assertEquals("P3", parsed9.get(parser.headers.get("priority")));
        Assert.assertEquals("maximo@amdocs.com", parsed9.get(parser.headers.get("creator")));
        Assert.assertEquals("vlukiyanov@amdocs.com", parsed9.get(parser.headers.get("assigned_to")));
        Assert.assertEquals("app-maxis-common", parsed9.get(parser.headers.get("component")));
        Assert.assertEquals("[]", parsed9.get(parser.headers.get("blocks")));
        Assert.assertEquals("[]", parsed9.get(parser.headers.get("depends_on")));
        Assert.assertEquals("", parsed9.get(parser.headers.get("dupe_of")));
        Assert.assertEquals(quote("[MAXIS][ASC] Request from customer to add RPL9-000006 fault code to template ID 504\\nBuild Name: \\nsee 1-SCO-6-009 NEXT1 BIT - Amdocs Service Control - UMB - IDD v2.0.docx in attach"), parsed9.get(parser.headers.get("description")));
        Assert.assertEquals("2018-11-05T09:23:39Z", parsed9.get(parser.headers.get("creation_time")));
        Assert.assertEquals("2018-11-06T13:23:57Z", parsed9.get(parser.headers.get("resolution_time")));
        Assert.assertEquals("28", parsed9.get(parser.headers.get("resolution_duration_hours")));
    }
}
