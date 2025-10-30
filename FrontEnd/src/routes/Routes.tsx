import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { BottomTabs } from "./bottom-tabs.routes";
import { ReportsProvider } from "../contexts/ReportContext";

export function Routes() {
  return (
    <ReportsProvider>
      <NavigationContainer>
        <BottomTabs />
      </NavigationContainer>
    </ReportsProvider>
  );
}
