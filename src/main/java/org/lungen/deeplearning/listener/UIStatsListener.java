package org.lungen.deeplearning.listener;

import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;

/**
 * StatsListener that reports stats to memory stats store,
 * and attaches it to UI Server.
 *
 */
public class UIStatsListener extends StatsListener {

    public UIStatsListener() {
        super(new InMemoryStatsStorage());
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach((InMemoryStatsStorage) getStorageRouter());
    }

}
