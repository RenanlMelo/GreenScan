import React from "react";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import History from "../screens/History/History";
import { Report } from "../screens/Report/Report";

export type HistoryStackParamList = {
  History: undefined;
  Report: { reportId: number };
};

const Stack = createNativeStackNavigator<HistoryStackParamList>();

export function HistoryStack() {
  return (
    <Stack.Navigator>
      <Stack.Screen
        name="History"
        component={History}
        options={{ title: "Histórico" }}
      />
      <Stack.Screen
        name="Report"
        component={Report}
        options={{ title: "Relatório" }}
      />
    </Stack.Navigator>
  );
}
