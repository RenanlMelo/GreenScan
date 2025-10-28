import { NavigationContainer } from "@react-navigation/native";
import { BottomTabs } from "./bottom-tabs.routes";

export function Routes() {
  return (
    <NavigationContainer>
      <BottomTabs />
    </NavigationContainer>
  );
}
