import { NavigationContainer } from "@react-navigation/native";
import { StackRoutes } from "./stack.routes";
import { BottomTabs } from "./bottom-tabs.routes";

export function Routes() {
  return (
    <NavigationContainer>
      {/* <StackRoutes /> */}
      <BottomTabs />
    </NavigationContainer>
  );
}
